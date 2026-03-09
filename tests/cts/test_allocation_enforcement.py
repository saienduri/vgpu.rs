"""
Allocation Enforcement Tests — Validates that the hip-limiter correctly enforces
memory limits configured via SHM for all allocation API variants.

Each test spawns a subprocess with LD_PRELOAD so the limiter initializes fresh.
The subprocess calls HIP APIs via hip_helper.py and prints results to stdout,
which the test parses to verify expected behavior.

Key behaviors under test:
  - Allocations within limits succeed (hipSuccess = 0)
  - Allocations exceeding limits return hipErrorOutOfMemory (code 2)
  - All allocation variants (hipMalloc, hipHostMalloc, etc.) are enforced
  - SHM pod_memory_used is updated correctly after allocations
"""

import pytest

from conftest import DEFAULT_TEST_UUID, MiB, GiB, parse_kv_output, requires_gpu
from shm_writer import DeviceSpec

pytestmark = requires_gpu

# hipErrorOutOfMemory
HIP_ERROR_OOM = 2


def test_alloc_within_limit(cts):
    """Allocate less than the configured limit — should succeed."""
    result = cts.run_hip_test("""
        from hip_helper import HIPRuntime, HIP_SUCCESS
        hip = HIPRuntime()
        err, ptr = hip.malloc_raw(1024 * 1024)  # 1 MiB, limit is 1 GiB
        print(f"err={err}")
        print(f"ptr={ptr}")
        if err == HIP_SUCCESS and ptr != 0:
            hip.free(ptr)
            print("PASS")
        else:
            print("FAIL")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Test failed: {result.stdout}"
    assert "err=0" in result.stdout


def test_alloc_at_limit(cts):
    """Allocate exactly the full limit — should succeed.

    The limiter checks `used + request > limit` (strictly greater than),
    so allocating exactly `limit` bytes when used=0 should pass.
    """
    result = cts.run_hip_test("""
        from hip_helper import HIPRuntime, HIP_SUCCESS
        hip = HIPRuntime()
        limit = 1024 * 1024 * 1024  # 1 GiB
        err, ptr = hip.malloc_raw(limit)
        print(f"err={err}")
        print(f"ptr={ptr}")
        if err == HIP_SUCCESS:
            print("PASS")
            hip.free(ptr)
        else:
            print(f"FAIL: err={err}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Expected success at limit: {result.stdout}"


def test_alloc_exceeds_limit(cts):
    """Allocate more than the limit — should return hipErrorOutOfMemory (2)."""
    result = cts.run_hip_test("""
        from hip_helper import HIPRuntime, HIP_ERROR_OUT_OF_MEMORY
        hip = HIPRuntime()
        limit = 1024 * 1024 * 1024  # 1 GiB
        request = limit + 1  # 1 byte over
        err, ptr = hip.malloc_raw(request)
        print(f"err={err}")
        if err == HIP_ERROR_OUT_OF_MEMORY:
            print("PASS")
        else:
            print(f"FAIL: expected err=2, got err={err}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Expected OOM: {result.stdout}"
    assert "err=2" in result.stdout


def test_incremental_fill(cts):
    """N small allocations summing to the limit should succeed; the N+1th should be denied."""
    result = cts.run_hip_test("""
        from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY

        hip = HIPRuntime()
        chunk = 64 * 1024 * 1024  # 64 MiB per chunk
        limit = 1024 * 1024 * 1024  # 1 GiB
        num_chunks = limit // chunk  # 16 chunks to fill

        ptrs = []
        all_ok = True
        for i in range(num_chunks):
            err, ptr = hip.malloc_raw(chunk)
            if err != HIP_SUCCESS:
                print(f"FAIL: chunk {i} failed with err={err}")
                all_ok = False
                break
            ptrs.append(ptr)

        if all_ok:
            # One more should fail
            err, ptr = hip.malloc_raw(chunk)
            if err == HIP_ERROR_OUT_OF_MEMORY:
                print("PASS")
            else:
                print(f"FAIL: extra chunk should be denied, got err={err}")

        # Clean up
        for p in ptrs:
            hip.free(p)
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Incremental fill test failed: {result.stdout}"


def test_alloc_after_free(cts):
    """Fill to limit, free some memory, then re-allocate — should succeed."""
    result = cts.run_hip_test("""
        from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY

        hip = HIPRuntime()
        limit = 1024 * 1024 * 1024  # 1 GiB
        half = limit // 2

        # Fill to limit with two halves
        err1, ptr1 = hip.malloc_raw(half)
        err2, ptr2 = hip.malloc_raw(half)
        assert err1 == HIP_SUCCESS and err2 == HIP_SUCCESS, f"Setup failed: {err1}, {err2}"

        # Verify we're at limit — next alloc should fail
        err_over, _ = hip.malloc_raw(half)
        assert err_over == HIP_ERROR_OUT_OF_MEMORY, f"Expected OOM, got {err_over}"

        # Free one half
        hip.free(ptr2)

        # Re-allocate the freed half — should succeed
        err3, ptr3 = hip.malloc_raw(half)
        if err3 == HIP_SUCCESS:
            print("PASS")
            hip.free(ptr3)
        else:
            print(f"FAIL: re-alloc after free failed with err={err3}")

        hip.free(ptr1)
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Alloc-after-free test failed: {result.stdout}"


ALLOC_VARIANT_SCRIPTS = {
    "hipMalloc": """\
from hip_helper import HIPRuntime, HIP_SUCCESS
hip = HIPRuntime()
err, ptr = hip.malloc_raw(1024 * 1024)
if err == HIP_SUCCESS:
    print("ALLOC_OK")
    hip.free(ptr)
else:
    print(f"ALLOC_FAIL={err}")
""",
    "hipExtMallocWithFlags": """\
from hip_helper import HIPRuntime
hip = HIPRuntime()
ptr = hip.ext_malloc_with_flags(1024 * 1024, 0)
print("ALLOC_OK")
hip.free(ptr)
""",
    "hipHostMalloc": """\
from hip_helper import HIPRuntime
hip = HIPRuntime()
ptr = hip.host_malloc(1024 * 1024, 0)
print("ALLOC_OK")
hip.host_free(ptr)
""",
    "hipMallocManaged": """\
from hip_helper import HIPRuntime
hip = HIPRuntime()
ptr = hip.malloc_managed(1024 * 1024, 1)
print("ALLOC_OK")
hip.free(ptr)
""",
    "hipMallocAsync": """\
from hip_helper import HIPRuntime
hip = HIPRuntime()
ptr = hip.malloc_async(1024 * 1024, 0)
print("ALLOC_OK")
hip.free_async(ptr, 0)
hip.device_synchronize()
""",
    "hipMallocFromPoolAsync": """\
from hip_helper import HIPRuntime
hip = HIPRuntime()
pool = hip.get_default_mem_pool(0)
ptr = hip.malloc_from_pool_async(1024 * 1024, pool, stream=0)
print("ALLOC_OK")
hip.free_async(ptr, 0)
hip.device_synchronize()
""",
}


@pytest.mark.parametrize("variant", list(ALLOC_VARIANT_SCRIPTS.keys()))
def test_each_alloc_variant(cts, variant):
    """Each allocation API variant should succeed within limits.

    Parametrized so each variant runs in its own subprocess, giving independent
    limiter init and a clear test name per variant in pytest output.
    """
    result = cts.run_hip_test(ALLOC_VARIANT_SCRIPTS[variant])
    assert result.succeeded, f"Subprocess failed for {variant}: {result.stderr}"
    assert "ALLOC_OK" in result.stdout, f"{variant} allocation failed: {result.stdout}"


def test_malloc_pitch(cts):
    """hipMallocPitch accounts for width*height bytes in the limiter.

    The Rust hook computes request_size = width * height before the limit check.
    """
    result = cts.run_hip_test("""
        from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY

        hip = HIPRuntime()
        limit = 1024 * 1024 * 1024  # 1 GiB

        # Allocate a pitched region within limits
        width = 1024
        height = 1024  # 1 MiB logical size
        ptr, pitch = hip.malloc_pitch(width, height)
        print(f"pitch={pitch}")
        print(f"ptr={ptr}")
        assert ptr != 0, "malloc_pitch returned null ptr"
        assert pitch >= width, f"pitch {pitch} < width {width}"
        hip.free(ptr)

        # Now try a pitched allocation that exceeds the limit
        # width * height > 1 GiB
        big_width = 1024 * 1024  # 1 MiB
        big_height = 1025  # total = 1 MiB * 1025 > 1 GiB
        try:
            ptr2, pitch2 = hip.malloc_pitch(big_width, big_height)
            # If it didn't raise, it means the native call succeeded but limiter should have blocked it
            print(f"FAIL: expected OOM for {big_width}x{big_height}")
            hip.free(ptr2)
        except Exception as exc:
            if "OutOfMemory" in str(exc) or "code 2" in str(exc):
                print("PASS")
            else:
                print(f"FAIL: unexpected error: {exc}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"malloc_pitch test failed: {result.stdout}"


def test_alloc_shm_accounting(cts):
    """Verify that SHM pod_memory_used is updated correctly after allocations."""
    alloc_size = 64 * MiB
    result = cts.run_hip_test(f"""
        from hip_helper import HIPRuntime, HIP_SUCCESS
        hip = HIPRuntime()
        size = {alloc_size}
        err, ptr = hip.malloc_raw(size)
        print(f"err={{err}}")
        print(f"ptr={{ptr}}")
        # Keep the allocation alive so we can read SHM from the parent
        if err == HIP_SUCCESS:
            # Read SHM from within the subprocess to report usage
            from shm_writer import read_pod_memory_used
            import os
            shm_path = os.environ["TF_SHM_FILE"]
            used = read_pod_memory_used(shm_path, 0)
            print(f"shm_used={{used}}")
            hip.free(ptr)
            used_after = read_pod_memory_used(shm_path, 0)
            print(f"shm_used_after_free={{used_after}}")
            print("PASS")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout
    # Verify SHM accounting values from subprocess output
    for line in result.stdout.splitlines():
        if line.startswith("shm_used="):
            used = int(line.split("=")[1])
            assert used == alloc_size, f"Expected SHM used={alloc_size}, got {used}"
        if line.startswith("shm_used_after_free="):
            used_after = int(line.split("=")[1])
            assert used_after == 0, f"Expected SHM used=0 after free, got {used_after}"


class TestMallocPitchAccounting:
    """hipMallocPitch accounting tracks pitch*height (actual GPU consumption).

    The GPU allocates pitch*height bytes where pitch >= width due to alignment.
    The limiter reserves width*height upfront, then adjusts to pitch*height after
    the native call returns the actual pitch. This prevents memory limit bypass
    via narrow pitched allocations with large alignment overhead.

    Gap: Previously tracked width*height which underreported actual GPU usage.
    """

    def test_malloc_pitch_tracks_pitch_times_height(self, cts):
        """Allocate with hipMallocPitch, verify pod_memory_used == pitch * height."""
        width = 1024
        height = 1024

        script = f"""\
import os
from hip_helper import HIPRuntime, HIP_SUCCESS
from shm_writer import read_pod_memory_used

hip = HIPRuntime()
shm_path = os.environ["TF_SHM_FILE"]

used_before = read_pod_memory_used(shm_path, 0)

ptr, pitch = hip.malloc_pitch({width}, {height})
print(f"PITCH={{pitch}}")
print(f"WIDTH={width}")
print(f"HEIGHT={height}")
print(f"PITCH_X_HEIGHT={{pitch * {height}}}")

used_after = read_pod_memory_used(shm_path, 0)
delta = used_after - used_before
print(f"SHM_DELTA={{delta}}")

hip.free(ptr)
print("DONE")
"""
        result = cts.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "DONE" in result.stdout, f"Script did not complete:\n{result.stdout}"

        values = parse_kv_output(result.stdout)
        shm_delta = values["SHM_DELTA"]
        pitch = values["PITCH"]
        pitch_x_height = values["PITCH_X_HEIGHT"]

        assert shm_delta == pitch_x_height, (
            f"SHM tracked {shm_delta} bytes but expected pitch*height={pitch_x_height}. "
            f"pitch={pitch}, width={width}. "
            f"The limiter should track pitch*height (actual GPU consumption)."
        )


class TestMallocPitchAlignmentOverhead:
    """hipMallocPitch alignment overhead can push an allocation over the limit.

    The limiter reserves width*height first, then after the native call discovers
    the actual pitch, tries to reserve the extra (pitch-width)*height. If that
    second reservation exceeds the limit, the limiter rolls back everything and
    frees the native allocation (mem.rs lines 203-213).

    This edge case is distinct from a simple over-limit pitch allocation — here,
    width*height fits within the limit but pitch*height does not.
    """

    def test_pitch_overhead_denied_when_over_limit(self, cts_factory):
        """Discover the GPU's actual pitch, then set a limit between
        width*height and pitch*height to trigger the alignment overhead denial."""

        # Step 1: Discover the actual pitch for a narrow width.
        # Use a raw (no-limiter) call to learn the pitch without enforcement.
        # We use a very large limit so the discovery allocation succeeds.
        discovery_fixture = cts_factory(
            devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=1 * GiB, device_idx=0)]
        )

        width = 128  # Narrow width — GPU will likely pad to 256 or 512
        height = 1024

        discover_script = f"""\
from hip_helper import HIPRuntime
hip = HIPRuntime()
ptr, pitch = hip.malloc_pitch({width}, {height})
print(f"PITCH={{pitch}}")
print(f"WIDTH={width}")
print(f"HEIGHT={height}")
print(f"ESTIMATED={{pitch * {height}}}")
hip.free(ptr)
"""
        result = discovery_fixture.run_hip_test(discover_script)
        assert result.succeeded, f"Discovery failed:\n{result.output}"

        values = parse_kv_output(result.stdout)
        pitch = values["PITCH"]
        actual_size = pitch * height
        estimated_size = width * height

        if pitch == width:
            pytest.skip(
                f"GPU did not pad width={width} (pitch==width), "
                f"cannot test alignment overhead path"
            )

        # Step 2: Set limit between width*height and pitch*height.
        # width*height passes initial reserve, but pitch*height exceeds limit.
        limit = estimated_size + (actual_size - estimated_size) // 2
        assert estimated_size <= limit < actual_size, (
            f"Limit {limit} not between estimated {estimated_size} and actual {actual_size}"
        )

        test_fixture = cts_factory(
            devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=limit, device_idx=0)]
        )

        test_script = f"""\
from hip_helper import HIPRuntime, HIP_ERROR_OUT_OF_MEMORY, HIPError
hip = HIPRuntime()
try:
    ptr, pitch = hip.malloc_pitch({width}, {height})
    # If we get here, the limiter did not catch the overhead
    print(f"UNEXPECTED_OK pitch={{pitch}}")
    hip.free(ptr)
except HIPError as exc:
    if exc.error_code == HIP_ERROR_OUT_OF_MEMORY:
        print("DENIED")
    else:
        print(f"UNEXPECTED={{exc.error_code}}")
"""
        result = test_fixture.run_hip_test(test_script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "DENIED" in result.stdout, (
            f"hipMallocPitch should be denied when pitch*height ({actual_size}) "
            f"exceeds limit ({limit}) even though width*height ({estimated_size}) "
            f"fits. Got:\n{result.stdout}"
        )

    def test_pitch_overhead_shm_unchanged_after_denial(self, cts_factory):
        """After the alignment overhead denial, SHM pod_memory_used must be 0.
        The limiter should fully roll back both the initial and overhead reservations."""
        discovery_fixture = cts_factory(
            devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=1 * GiB, device_idx=0)]
        )

        width = 128
        height = 1024

        discover_script = f"""\
from hip_helper import HIPRuntime
hip = HIPRuntime()
ptr, pitch = hip.malloc_pitch({width}, {height})
print(f"PITCH={{pitch}}")
hip.free(ptr)
"""
        result = discovery_fixture.run_hip_test(discover_script)
        assert result.succeeded, f"Discovery failed:\n{result.output}"

        values = parse_kv_output(result.stdout)
        pitch = values["PITCH"]
        actual_size = pitch * height
        estimated_size = width * height

        if pitch == width:
            pytest.skip("GPU did not pad width, cannot test alignment overhead path")

        limit = estimated_size + (actual_size - estimated_size) // 2

        test_fixture = cts_factory(
            devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=limit, device_idx=0)]
        )

        test_script = f"""\
from hip_helper import HIPRuntime, HIPError
hip = HIPRuntime()
try:
    ptr, pitch = hip.malloc_pitch({width}, {height})
    hip.free(ptr)
except HIPError:
    pass
print("DONE")
"""
        result = test_fixture.run_hip_test(test_script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"

        shm_used = test_fixture.read_pod_memory_used(device_idx=0)
        assert shm_used == 0, (
            f"SHM pod_memory_used ({shm_used}) should be 0 after pitch overhead "
            f"denial — the initial reservation must be fully rolled back"
        )


class TestPerVariantOomEnforcement:
    """Each allocation variant must independently enforce the memory limit.

    Gap: Previously only hipMalloc was tested for OOM. A bug where hipHostMalloc
    or hipMallocManaged skipped enforcement would go undetected.
    """

    VARIANT_SCRIPTS = {
        "hipMalloc": """\
from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY
hip = HIPRuntime()

# Fill most of the limit with hipMalloc
fill_size = {fill_size}
err, fill_ptr = hip.malloc_raw(fill_size)
assert err == HIP_SUCCESS, f"Fill alloc failed: {{err}}"

# Now try to allocate more than remains — should be denied
over_size = {over_size}
err_over, _ = hip.malloc_raw(over_size)
if err_over == HIP_ERROR_OUT_OF_MEMORY:
    print("DENIED")
else:
    print(f"UNEXPECTED={{err_over}}")

hip.free(fill_ptr)
""",
        "hipHostMalloc": """\
from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY, HIPError
hip = HIPRuntime()

fill_size = {fill_size}
err, fill_ptr = hip.malloc_raw(fill_size)
assert err == HIP_SUCCESS, f"Fill alloc failed: {{err}}"

over_size = {over_size}
try:
    ptr = hip.host_malloc(over_size, 0)
    print("UNEXPECTED=0")
    hip.host_free(ptr)
except HIPError as exc:
    if exc.error_code == HIP_ERROR_OUT_OF_MEMORY:
        print("DENIED")
    else:
        print(f"UNEXPECTED={{exc.error_code}}")

hip.free(fill_ptr)
""",
        "hipExtMallocWithFlags": """\
from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY, HIPError
hip = HIPRuntime()

fill_size = {fill_size}
err, fill_ptr = hip.malloc_raw(fill_size)
assert err == HIP_SUCCESS, f"Fill alloc failed: {{err}}"

over_size = {over_size}
try:
    ptr = hip.ext_malloc_with_flags(over_size, 0)
    print("UNEXPECTED=0")
    hip.free(ptr)
except HIPError as exc:
    if exc.error_code == HIP_ERROR_OUT_OF_MEMORY:
        print("DENIED")
    else:
        print(f"UNEXPECTED={{exc.error_code}}")

hip.free(fill_ptr)
""",
        "hipMallocManaged": """\
from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY, HIPError
hip = HIPRuntime()

fill_size = {fill_size}
err, fill_ptr = hip.malloc_raw(fill_size)
assert err == HIP_SUCCESS, f"Fill alloc failed: {{err}}"

over_size = {over_size}
try:
    ptr = hip.malloc_managed(over_size, 1)
    print("UNEXPECTED=0")
    hip.free(ptr)
except HIPError as exc:
    if exc.error_code == HIP_ERROR_OUT_OF_MEMORY:
        print("DENIED")
    else:
        print(f"UNEXPECTED={{exc.error_code}}")

hip.free(fill_ptr)
""",
        "hipMallocAsync": """\
from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY, HIPError
hip = HIPRuntime()

fill_size = {fill_size}
err, fill_ptr = hip.malloc_raw(fill_size)
assert err == HIP_SUCCESS, f"Fill alloc failed: {{err}}"

over_size = {over_size}
try:
    ptr = hip.malloc_async(over_size, 0)
    print("UNEXPECTED=0")
    hip.free_async(ptr, 0)
    hip.device_synchronize()
except HIPError as exc:
    if exc.error_code == HIP_ERROR_OUT_OF_MEMORY:
        print("DENIED")
    else:
        print(f"UNEXPECTED={{exc.error_code}}")

hip.free(fill_ptr)
""",
        "hipMallocFromPoolAsync": """\
from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY, HIPError
hip = HIPRuntime()

fill_size = {fill_size}
err, fill_ptr = hip.malloc_raw(fill_size)
assert err == HIP_SUCCESS, f"Fill alloc failed: {{err}}"

pool = hip.get_default_mem_pool(0)
over_size = {over_size}
try:
    ptr = hip.malloc_from_pool_async(over_size, pool, stream=0)
    print("UNEXPECTED=0")
    hip.free_async(ptr, 0)
    hip.device_synchronize()
except HIPError as exc:
    if exc.error_code == HIP_ERROR_OUT_OF_MEMORY:
        print("DENIED")
    else:
        print(f"UNEXPECTED={{exc.error_code}}")

hip.free(fill_ptr)
""",
        "hipMallocPitch": """\
from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY, HIPError
hip = HIPRuntime()

fill_size = {fill_size}
err, fill_ptr = hip.malloc_raw(fill_size)
assert err == HIP_SUCCESS, f"Fill alloc failed: {{err}}"

# width * height = over_size, which when added to fill_size exceeds limit.
# Use width=over_size, height=1 so the logical size is exactly over_size.
over_size = {over_size}
try:
    ptr, pitch = hip.malloc_pitch(over_size, 1)
    print("UNEXPECTED=0")
    hip.free(ptr)
except HIPError as exc:
    if exc.error_code == HIP_ERROR_OUT_OF_MEMORY:
        print("DENIED")
    else:
        print(f"UNEXPECTED={{exc.error_code}}")

hip.free(fill_ptr)
""",
    }

    @pytest.mark.parametrize("variant", list(VARIANT_SCRIPTS.keys()))
    def test_variant_oom(self, cts_factory, variant):
        """Allocate near the limit, then verify the specific variant is denied."""
        mem_limit = 4 * MiB
        fixture = cts_factory(
            devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=mem_limit, device_idx=0)]
        )
        fill_size = 3 * MiB
        over_size = 2 * MiB  # fill_size + over_size > mem_limit

        script = self.VARIANT_SCRIPTS[variant].format(
            fill_size=fill_size, over_size=over_size,
        )
        result = fixture.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed for {variant}: {result.stderr}"
        assert "DENIED" in result.stdout, (
            f"{variant} should be denied when exceeding limit, got: {result.stdout}"
        )
