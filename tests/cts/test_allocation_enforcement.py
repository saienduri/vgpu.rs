"""
Allocation Enforcement Tests — Validates that the hip-limiter correctly enforces
memory limits configured via SHM for all allocation API variants.

Each test spawns a subprocess with LD_PRELOAD so the limiter initializes fresh.
The subprocess calls HIP APIs via hip_helper.py and prints results to stdout,
which the test parses to verify expected behavior.

Key behaviors under test:
  - Allocations within limits succeed (hipSuccess = 0)
  - Allocations exceeding limits return hipErrorOutOfMemory (code 2)
  - All allocation variants (hipMalloc, hipMallocManaged, etc.) are enforced
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
    "hipMallocPitch": """\
from hip_helper import HIPRuntime
hip = HIPRuntime()
ptr, pitch = hip.malloc_pitch(1024, 1024)
print("ALLOC_OK")
hip.free(ptr)
""",
    "hipMemAllocPitch": """\
from hip_helper import HIPRuntime
hip = HIPRuntime()
ptr, pitch = hip.mem_alloc_pitch(1024, 1024)
print("ALLOC_OK")
hip.free(ptr)
""",
    "hipMalloc3D": """\
from hip_helper import HIPRuntime
hip = HIPRuntime()
ptr, pitch, xsize, ysize = hip.malloc_3d(1024, 1024, 1)
print("ALLOC_OK")
hip.free(ptr)
""",
    "hipMemCreate": """\
from hip_helper import HIPRuntime
hip = HIPRuntime()
granularity = hip.get_allocation_granularity(0)
size = ((1024 * 1024 + granularity - 1) // granularity) * granularity
handle = hip.mem_create(size, 0)
print("ALLOC_OK")
hip.mem_release(handle)
""",
    "hipMallocArray": """\
from hip_helper import HIPRuntime, HIPError, HIP_ERROR_NOT_SUPPORTED
hip = HIPRuntime()
try:
    arr = hip.malloc_array(width=256, height=256, desc_x=32)
    print("ALLOC_OK")
    hip.free_array(arr)
except HIPError as e:
    if e.error_code == HIP_ERROR_NOT_SUPPORTED:
        print("NOT_SUPPORTED")
    else:
        raise
""",
    "hipMalloc3DArray": """\
from hip_helper import HIPRuntime, HIPError, HIP_ERROR_NOT_SUPPORTED
hip = HIPRuntime()
try:
    arr = hip.malloc_3d_array(width=64, height=64, depth=64, desc_x=32)
    print("ALLOC_OK")
    hip.free_array(arr)
except HIPError as e:
    if e.error_code == HIP_ERROR_NOT_SUPPORTED:
        print("NOT_SUPPORTED")
    else:
        raise
""",
    "hipArrayCreate": """\
from hip_helper import HIPRuntime, HIPError, HIP_AD_FORMAT_FLOAT, HIP_ERROR_NOT_SUPPORTED
hip = HIPRuntime()
try:
    arr = hip.array_create(width=256, height=256, fmt=HIP_AD_FORMAT_FLOAT, num_channels=1)
    print("ALLOC_OK")
    hip.array_destroy(arr)
except HIPError as e:
    if e.error_code == HIP_ERROR_NOT_SUPPORTED:
        print("NOT_SUPPORTED")
    else:
        raise
""",
    "hipArray3DCreate": """\
from hip_helper import HIPRuntime, HIPError, HIP_AD_FORMAT_FLOAT, HIP_ERROR_NOT_SUPPORTED
hip = HIPRuntime()
try:
    arr = hip.array_3d_create(width=64, height=64, depth=64, fmt=HIP_AD_FORMAT_FLOAT, num_channels=1)
    print("ALLOC_OK")
    hip.array_destroy(arr)
except HIPError as e:
    if e.error_code == HIP_ERROR_NOT_SUPPORTED:
        print("NOT_SUPPORTED")
    else:
        raise
""",
    "hipMallocMipmappedArray": """\
from hip_helper import HIPRuntime, HIPError, HIP_ERROR_NOT_SUPPORTED
hip = HIPRuntime()
try:
    arr = hip.malloc_mipmapped_array(width=256, height=256, num_levels=4, desc_x=32)
    print("ALLOC_OK")
    hip.free_mipmapped_array(arr)
except HIPError as e:
    if e.error_code == HIP_ERROR_NOT_SUPPORTED:
        print("NOT_SUPPORTED")
    else:
        raise
""",
    "hipMipmappedArrayCreate": """\
from hip_helper import HIPRuntime, HIPError, HIP_AD_FORMAT_FLOAT, HIP_ERROR_NOT_SUPPORTED
hip = HIPRuntime()
try:
    arr = hip.mipmapped_array_create(width=256, height=256, num_levels=4, fmt=HIP_AD_FORMAT_FLOAT, num_channels=1)
    print("ALLOC_OK")
    hip.mipmapped_array_destroy(arr)
except HIPError as e:
    if e.error_code == HIP_ERROR_NOT_SUPPORTED:
        print("NOT_SUPPORTED")
    else:
        raise
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
    if "NOT_SUPPORTED" in result.stdout:
        pytest.skip(f"{variant} not supported on this GPU")
    assert "ALLOC_OK" in result.stdout, f"{variant} allocation failed: {result.stdout}"


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


def _assert_pitched_2d_accounting(cts, alloc_call, api_name, width, height):
    """Verify SHM pod_memory_used == pitch * height after a 2D pitched allocation.

    Used by both hipMallocPitch and hipMemAllocPitch accounting tests.
    alloc_call: format string with {width} and {height} placeholders that sets ptr, pitch.
    """
    script = f"""\
import os
from hip_helper import HIPRuntime, HIP_SUCCESS
from shm_writer import read_pod_memory_used

hip = HIPRuntime()
shm_path = os.environ["TF_SHM_FILE"]

used_before = read_pod_memory_used(shm_path, 0)

{alloc_call.format(width=width, height=height)}
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
    pitch_x_height = values["PITCH_X_HEIGHT"]

    assert shm_delta == pitch_x_height, (
        f"{api_name}: SHM tracked {shm_delta} bytes but expected pitch*height={pitch_x_height}. "
        f"pitch={values['PITCH']}, width={width}. "
        f"The limiter should track pitch*height (actual GPU consumption)."
    )


class TestMallocPitchAccounting:
    """hipMallocPitch accounting tracks pitch*height (actual GPU consumption)."""

    def test_malloc_pitch_tracks_pitch_times_height(self, cts):
        _assert_pitched_2d_accounting(
            cts, _PITCH_ALLOC_2D, "hipMallocPitch", width=1024, height=1024,
        )


class TestMemAllocPitchAccounting:
    """hipMemAllocPitch (driver API) accounting tracks pitch*height (actual GPU consumption)."""

    def test_mem_alloc_pitch_tracks_pitch_times_height(self, cts):
        _assert_pitched_2d_accounting(
            cts, _PITCH_ALLOC_2D_DRIVER, "hipMemAllocPitch", width=1024, height=1024,
        )


def _discover_pitch(cts_factory, alloc_call, width, height, depth=None):
    """Discover the GPU's actual pitch for a given allocation shape.

    Returns (pitch, estimated_size, actual_size) or calls pytest.skip if pitch == width.
    """
    discovery_fixture = cts_factory(
        devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=1 * GiB, device_idx=0)]
    )

    discover_script = f"""\
from hip_helper import HIPRuntime
hip = HIPRuntime()
{alloc_call}
print(f"PITCH={{pitch}}")
hip.free(ptr)
"""
    result = discovery_fixture.run_hip_test(discover_script)
    assert result.succeeded, f"Discovery failed:\n{result.output}"

    values = parse_kv_output(result.stdout)
    pitch = values["PITCH"]

    dims = [pitch, height] + ([depth] if depth else [])
    actual_size = 1
    for d in dims:
        actual_size *= d

    est_dims = [width, height] + ([depth] if depth else [])
    estimated_size = 1
    for d in est_dims:
        estimated_size *= d

    if pitch == width:
        pytest.skip(
            f"GPU did not pad width={width} (pitch==width), "
            f"cannot test alignment overhead path"
        )

    return pitch, estimated_size, actual_size


def _assert_overhead_denied(cts_factory, alloc_call, api_name, width, height, depth=None):
    """Assert that alignment overhead pushes an allocation over the limit.

    Discovers pitch, sets limit between estimated and actual, verifies OOM denial.
    """
    pitch, estimated_size, actual_size = _discover_pitch(
        cts_factory, alloc_call, width, height, depth
    )
    limit = estimated_size + (actual_size - estimated_size) // 2
    assert estimated_size <= limit < actual_size

    test_fixture = cts_factory(
        devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=limit, device_idx=0)]
    )

    test_script = f"""\
from hip_helper import HIPRuntime, HIP_ERROR_OUT_OF_MEMORY, HIPError
hip = HIPRuntime()
try:
    {alloc_call}
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
        f"{api_name} should be denied when actual ({actual_size}) "
        f"exceeds limit ({limit}) even though estimated ({estimated_size}) "
        f"fits. Got:\n{result.stdout}"
    )


def _assert_overhead_shm_unchanged(cts_factory, alloc_call, api_name, width, height, depth=None):
    """Assert that SHM pod_memory_used is 0 after alignment overhead denial."""
    pitch, estimated_size, actual_size = _discover_pitch(
        cts_factory, alloc_call, width, height, depth
    )
    limit = estimated_size + (actual_size - estimated_size) // 2

    test_fixture = cts_factory(
        devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=limit, device_idx=0)]
    )

    test_script = f"""\
from hip_helper import HIPRuntime, HIPError
hip = HIPRuntime()
try:
    {alloc_call}
    hip.free(ptr)
except HIPError:
    pass
print("DONE")
"""
    result = test_fixture.run_hip_test(test_script)
    assert result.succeeded, f"Subprocess failed:\n{result.output}"

    shm_used = test_fixture.read_pod_memory_used(device_idx=0)
    assert shm_used == 0, (
        f"SHM pod_memory_used ({shm_used}) should be 0 after {api_name} "
        f"overhead denial — the reservation must be fully rolled back"
    )


# Pitched alloc calls used by discovery and denial helpers.
# Each must set `ptr` and `pitch` variables in the subprocess script scope.
_PITCH_ALLOC_2D = "ptr, pitch = hip.malloc_pitch({width}, {height})"
_PITCH_ALLOC_2D_DRIVER = "ptr, pitch = hip.mem_alloc_pitch({width}, {height})"
_PITCH_ALLOC_3D = "ptr, pitch, xsize, ysize = hip.malloc_3d({width}, {height}, {depth})"


class TestMallocPitchAlignmentOverhead:
    """hipMallocPitch alignment overhead can push an allocation over the limit."""

    WIDTH, HEIGHT = 128, 1024

    def test_pitch_overhead_denied_when_over_limit(self, cts_factory):
        _assert_overhead_denied(
            cts_factory,
            _PITCH_ALLOC_2D.format(width=self.WIDTH, height=self.HEIGHT),
            "hipMallocPitch", self.WIDTH, self.HEIGHT,
        )

    def test_pitch_overhead_shm_unchanged_after_denial(self, cts_factory):
        _assert_overhead_shm_unchanged(
            cts_factory,
            _PITCH_ALLOC_2D.format(width=self.WIDTH, height=self.HEIGHT),
            "hipMallocPitch", self.WIDTH, self.HEIGHT,
        )


class TestMemAllocPitchAlignmentOverhead:
    """hipMemAllocPitch (driver API) alignment overhead can push an allocation over the limit."""

    WIDTH, HEIGHT = 128, 1024

    def test_pitch_overhead_denied_when_over_limit(self, cts_factory):
        _assert_overhead_denied(
            cts_factory,
            _PITCH_ALLOC_2D_DRIVER.format(width=self.WIDTH, height=self.HEIGHT),
            "hipMemAllocPitch", self.WIDTH, self.HEIGHT,
        )

    def test_pitch_overhead_shm_unchanged_after_denial(self, cts_factory):
        _assert_overhead_shm_unchanged(
            cts_factory,
            _PITCH_ALLOC_2D_DRIVER.format(width=self.WIDTH, height=self.HEIGHT),
            "hipMemAllocPitch", self.WIDTH, self.HEIGHT,
        )


class TestMalloc3DAccounting:
    """hipMalloc3D accounting tracks pitch * height * depth (actual GPU consumption).

    Like hipMallocPitch, hipMalloc3D returns a pitched pointer where pitch >= width
    due to alignment. The limiter must account for pitch*height*depth, not
    width*height*depth, to avoid under-reporting actual GPU memory usage.
    """

    def test_malloc_3d_basic(self, cts):
        """hipMalloc3D with known dimensions should succeed and return valid pitched pointer."""
        result = cts.run_hip_test("""\
from hip_helper import HIPRuntime
hip = HIPRuntime()

ptr, pitch, xsize, ysize = hip.malloc_3d(1024, 1024, 1)
print(f"PTR={ptr}")
print(f"PITCH={pitch}")
print(f"XSIZE={xsize}")
print(f"YSIZE={ysize}")
assert ptr != 0, "hipMalloc3D returned null ptr"
assert pitch >= 1024, f"pitch {pitch} < width 1024"
hip.free(ptr)
print("ALLOC_OK")
""")
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "ALLOC_OK" in result.stdout, f"hipMalloc3D basic test failed:\n{result.stdout}"

    def test_malloc_3d_tracks_pitch_times_height_times_depth(self, cts):
        """Allocate with hipMalloc3D, verify pod_memory_used == pitch * height * depth."""
        width = 1024
        height = 512
        depth = 2

        script = f"""\
import os
from hip_helper import HIPRuntime, HIP_SUCCESS
from shm_writer import read_pod_memory_used

hip = HIPRuntime()
shm_path = os.environ["TF_SHM_FILE"]

used_before = read_pod_memory_used(shm_path, 0)

ptr, pitch, xsize, ysize = hip.malloc_3d({width}, {height}, {depth})
print(f"PITCH={{pitch}}")
print(f"WIDTH={width}")
print(f"HEIGHT={height}")
print(f"DEPTH={depth}")
print(f"PITCH_X_HEIGHT_X_DEPTH={{pitch * {height} * {depth}}}")

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
        expected = pitch * height * depth

        assert shm_delta == expected, (
            f"SHM tracked {shm_delta} bytes but expected pitch*height*depth={expected}. "
            f"pitch={pitch}, width={width}, height={height}, depth={depth}. "
            f"The limiter should track pitch*height*depth (actual GPU consumption)."
        )

    def test_malloc_3d_multi_depth(self, cts):
        """hipMalloc3D with depth > 1 should account for all depth slices."""
        width = 256
        height = 256
        depth = 4

        script = f"""\
import os
from hip_helper import HIPRuntime
from shm_writer import read_pod_memory_used

hip = HIPRuntime()
shm_path = os.environ["TF_SHM_FILE"]

used_before = read_pod_memory_used(shm_path, 0)

ptr, pitch, xsize, ysize = hip.malloc_3d({width}, {height}, {depth})
print(f"PITCH={{pitch}}")

used_after = read_pod_memory_used(shm_path, 0)
delta = used_after - used_before
print(f"SHM_DELTA={{delta}}")
print(f"EXPECTED={{pitch * {height} * {depth}}}")

hip.free(ptr)
print("DONE")
"""
        result = cts.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "DONE" in result.stdout, f"Script did not complete:\n{result.stdout}"

        values = parse_kv_output(result.stdout)
        shm_delta = values["SHM_DELTA"]
        expected = values["EXPECTED"]

        assert shm_delta == expected, (
            f"SHM tracked {shm_delta} bytes but expected {expected} "
            f"(pitch * height * depth with depth={depth})."
        )


class TestMemCreateAccounting:
    """hipMemCreate/hipMemRelease accounting via opaque handles."""

    def test_mem_create_tracks_size(self, cts):
        """hipMemCreate should track the requested size in SHM accounting."""
        result = cts.run_hip_test("""\
import os
from hip_helper import HIPRuntime
from shm_writer import read_pod_memory_used

hip = HIPRuntime()
shm_path = os.environ["TF_SHM_FILE"]

granularity = hip.get_allocation_granularity(0)
alloc_size = ((16 * 1024 * 1024 + granularity - 1) // granularity) * granularity

used_before = read_pod_memory_used(shm_path, 0)
handle = hip.mem_create(alloc_size, 0)
used_after_alloc = read_pod_memory_used(shm_path, 0)
hip.mem_release(handle)
used_after_free = read_pod_memory_used(shm_path, 0)

delta_alloc = used_after_alloc - used_before
delta_free = used_after_alloc - used_after_free

print(f"alloc_size={alloc_size}")
print(f"delta_alloc={delta_alloc}")
print(f"delta_free={delta_free}")

if delta_alloc == alloc_size and delta_free == alloc_size:
    print("PASS")
else:
    print(f"FAIL: delta_alloc={delta_alloc} delta_free={delta_free} expected={alloc_size}")
""")
        assert result.succeeded, f"Subprocess failed: {result.stderr}"
        assert "PASS" in result.stdout, f"hipMemCreate accounting failed: {result.stdout}"


class TestMalloc3DAlignmentOverhead:
    """hipMalloc3D alignment overhead can push an allocation over the limit."""

    WIDTH, HEIGHT, DEPTH = 128, 1024, 2

    def test_3d_overhead_denied_when_over_limit(self, cts_factory):
        _assert_overhead_denied(
            cts_factory,
            _PITCH_ALLOC_3D.format(width=self.WIDTH, height=self.HEIGHT, depth=self.DEPTH),
            "hipMalloc3D", self.WIDTH, self.HEIGHT, self.DEPTH,
        )

    def test_3d_overhead_shm_unchanged_after_denial(self, cts_factory):
        _assert_overhead_shm_unchanged(
            cts_factory,
            _PITCH_ALLOC_3D.format(width=self.WIDTH, height=self.HEIGHT, depth=self.DEPTH),
            "hipMalloc3D", self.WIDTH, self.HEIGHT, self.DEPTH,
        )


# Template for the common exception-based OOM pattern: fill near limit with
# hipMalloc, then attempt the variant alloc and expect HIP_ERROR_OUT_OF_MEMORY.
# Used by TestPerVariantOomEnforcement below.
_OOM_TEMPLATE = """\
from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY, HIPError
hip = HIPRuntime()

fill_size = {{fill_size}}
err, fill_ptr = hip.malloc_raw(fill_size)
assert err == HIP_SUCCESS, f"Fill alloc failed: {{{{err}}}}"

{preamble}over_size = {{over_size}}
try:
    {alloc_expr}
    print("UNEXPECTED=0")
    {free_expr}
except HIPError as exc:
    if exc.error_code == HIP_ERROR_OUT_OF_MEMORY:
        print("DENIED")
    else:
        print(f"UNEXPECTED={{{{exc.error_code}}}}")

hip.free(fill_ptr)
"""


def _oom_script(alloc_expr, free_expr, preamble=""):
    """Build an OOM test script from the template.

    Two-stage format: this call resolves {alloc_expr}/{free_expr}/{preamble},
    leaving {{fill_size}}/{{over_size}} as {fill_size}/{over_size} for the
    test method's .format() call.
    """
    return _OOM_TEMPLATE.format(
        alloc_expr=alloc_expr, free_expr=free_expr,
        preamble=preamble + "\n" if preamble else "",
    )


class TestPerVariantOomEnforcement:
    """Each allocation variant must independently enforce the memory limit.

    Gap: Previously only hipMalloc was tested for OOM. A bug where hipMallocManaged
    or hipMallocAsync skipped enforcement would go undetected.
    """

    VARIANT_SCRIPTS = {
        # hipMalloc uses raw error codes (no exception), so it has a unique script.
        "hipMalloc": """\
from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY
hip = HIPRuntime()

fill_size = {fill_size}
err, fill_ptr = hip.malloc_raw(fill_size)
assert err == HIP_SUCCESS, f"Fill alloc failed: {{err}}"

over_size = {over_size}
err_over, _ = hip.malloc_raw(over_size)
if err_over == HIP_ERROR_OUT_OF_MEMORY:
    print("DENIED")
else:
    print(f"UNEXPECTED={{err_over}}")

hip.free(fill_ptr)
""",
        # Standard exception-based variants — all share the same template.
        "hipExtMallocWithFlags": _oom_script(
            "ptr = hip.ext_malloc_with_flags(over_size, 0)", "hip.free(ptr)"),
        "hipMallocManaged": _oom_script(
            "ptr = hip.malloc_managed(over_size, 1)", "hip.free(ptr)"),
        # Async variants need device_synchronize for cleanup.
        "hipMallocAsync": _oom_script(
            "ptr = hip.malloc_async(over_size, 0)",
            "hip.free_async(ptr, 0)\n    hip.device_synchronize()"),
        "hipMallocFromPoolAsync": _oom_script(
            "ptr = hip.malloc_from_pool_async(over_size, pool, stream=0)",
            "hip.free_async(ptr, 0)\n    hip.device_synchronize()",
            preamble="pool = hip.get_default_mem_pool(0)"),
        # Pitched variants use width=over_size, height=1 [, depth=1] so logical
        # size is exactly over_size. This tests the estimated-size denial path;
        # alignment overhead denial is covered by TestMallocPitch/3DAlignmentOverhead.
        "hipMallocPitch": _oom_script(
            "ptr, pitch = hip.malloc_pitch(over_size, 1)", "hip.free(ptr)"),
        "hipMemAllocPitch": _oom_script(
            "ptr, pitch = hip.mem_alloc_pitch(over_size, 1)", "hip.free(ptr)"),
        "hipMalloc3D": _oom_script(
            "ptr, pitch, xsize, ysize = hip.malloc_3d(over_size, 1, 1)", "hip.free(ptr)"),
        "hipMemCreate": _oom_script(
            "aligned = ((over_size + granularity - 1) // granularity) * granularity\n"
            "    handle = hip.mem_create(aligned, 0)",
            "hip.mem_release(handle)",
            preamble="granularity = hip.get_allocation_granularity(0)"),
        "hipMallocArray": _oom_script(
            "arr = hip.malloc_array(width=over_size // 4, height=1, desc_x=32)",
            "hip.free_array(arr)"),
        "hipMalloc3DArray": _oom_script(
            "arr = hip.malloc_3d_array(width=over_size // 4, height=1, depth=1, desc_x=32)",
            "hip.free_array(arr)"),
        "hipArrayCreate": _oom_script(
            "arr = hip.array_create(width=over_size // 4, height=1)",
            "hip.array_destroy(arr)"),
        "hipArray3DCreate": _oom_script(
            "arr = hip.array_3d_create(width=over_size // 4, height=1, depth=1)",
            "hip.array_destroy(arr)"),
        "hipMallocMipmappedArray": _oom_script(
            "arr = hip.malloc_mipmapped_array(width=over_size // 4, height=1, num_levels=1, desc_x=32)",
            "hip.free_mipmapped_array(arr)"),
        "hipMipmappedArrayCreate": _oom_script(
            "arr = hip.mipmapped_array_create(width=over_size // 4, height=1, num_levels=1)",
            "hip.mipmapped_array_destroy(arr)"),
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
        if "UNEXPECTED=801" in result.stdout:
            pytest.skip(f"{variant} not supported on this GPU")
        assert "DENIED" in result.stdout, (
            f"{variant} should be denied when exceeding limit, got: {result.stdout}"
        )
