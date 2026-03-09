"""
Edge Case Tests — Validates limiter behavior at boundaries and unusual conditions.

Key behaviors under test:
  - Zero-size allocation: passes through, not tracked in accounting
  - Very large allocation (near u64::MAX): denied without overflow
  - Rapid alloc/free cycles: no accounting leaks after many iterations
  - Missing SHM file: graceful passthrough (no crash, native HIP calls work)
"""

import pytest

from conftest import DEFAULT_TEST_UUID, MiB, GiB, requires_gpu
from shm_writer import DeviceSpec

pytestmark = requires_gpu

HIP_ERROR_OOM = 2


def test_zero_size_alloc(cts):
    """hipMalloc(0) should succeed and not be tracked in SHM accounting.

    The Rust limiter's record_allocation returns early if size==0, so
    pod_memory_used should not change. The check_and_alloc macro also
    passes since 0 + 0 <= limit.
    """
    result = cts.run_hip_test("""
        import os
        from hip_helper import HIPRuntime, HIP_SUCCESS
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]

        used_before = read_pod_memory_used(shm_path, 0)
        err, ptr = hip.malloc_raw(0)
        used_after = read_pod_memory_used(shm_path, 0)

        print(f"err={err}")
        print(f"ptr={ptr}")
        print(f"used_before={used_before}")
        print(f"used_after={used_after}")

        # hipMalloc(0) should succeed
        err_ok = (err == HIP_SUCCESS)
        # Accounting should not change
        acct_ok = (used_before == used_after)

        if err_ok and acct_ok:
            print("PASS")
            # Clean up — free even zero-size ptr (safe, limiter won't find it in tracker)
            if ptr != 0:
                hip.free(ptr)
        else:
            print(f"FAIL: err_ok={err_ok} acct_ok={acct_ok}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Zero-size alloc test failed: {result.stdout}"


def test_very_large_alloc(cts):
    """Attempting to allocate near u64::MAX should return OOM without overflow.

    The Rust limiter uses saturating_add: used.saturating_add(request_size) > mem_limit.
    When request_size is near u64::MAX, saturating_add clamps to u64::MAX, which is
    always > mem_limit, so the allocation is correctly denied.
    """
    result = cts.run_hip_test("""
        from hip_helper import HIPRuntime, HIP_ERROR_OUT_OF_MEMORY

        hip = HIPRuntime()

        # Near u64::MAX — this should be denied by the limiter's saturating_add check
        # We use a large but representable Python int. ctypes c_size_t will wrap on
        # overflow, but even 2^63-1 is far above any mem_limit.
        huge_size = (1 << 63) - 1  # 9.2 exabytes

        err, ptr = hip.malloc_raw(huge_size)
        print(f"err={err}")

        if err == HIP_ERROR_OUT_OF_MEMORY:
            print("PASS")
        else:
            print(f"FAIL: expected OOM (err=2), got err={err}")
            if ptr != 0:
                hip.free(ptr)
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Very large alloc test failed: {result.stdout}"


def test_rapid_alloc_free_loop(cts):
    """10,000 alloc/free cycles should leave SHM accounting at zero (no leaks).

    This stress-tests that record_allocation and record_free are balanced
    across many iterations, and no pointer tracking entries are leaked.
    """
    result = cts.run_hip_test("""
        import os
        from hip_helper import HIPRuntime, HIP_SUCCESS
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]
        iterations = 10000
        alloc_size = 4096  # 4 KiB per iteration

        used_before = read_pod_memory_used(shm_path, 0)

        for i in range(iterations):
            err, ptr = hip.malloc_raw(alloc_size)
            if err != HIP_SUCCESS:
                print(f"FAIL: alloc failed at iteration {i}, err={err}")
                break
            hip.free(ptr)
        else:
            # All iterations completed
            used_after = read_pod_memory_used(shm_path, 0)
            print(f"used_before={used_before}")
            print(f"used_after={used_after}")
            print(f"iterations={iterations}")

            if used_after == used_before:
                print("PASS")
            else:
                leak = used_after - used_before
                print(f"FAIL: accounting leak of {leak} bytes after {iterations} cycles")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Rapid alloc/free test failed: {result.stdout}"


class TestAllocWithNoShm:
    """Allocation without SHM file — limiter should fall back to native HIP.

    Gap: Previously this test accepted any outcome, making it impossible to fail.
    The expected behavior is: limiter cannot access SHM, falls through to native
    HIP allocator, allocation succeeds but is untracked.
    """

    def test_alloc_with_no_shm(self, cts_factory):
        """Without a valid SHM file, the limiter should fall back to the native
        allocator. The allocation should succeed but not be tracked in SHM."""
        cts = cts_factory(
            devices=[
                DeviceSpec(
                    uuid=DEFAULT_TEST_UUID,
                    mem_limit=GiB,
                    device_idx=0,
                )
            ]
        )

        result = cts.run_hip_test(
            """
            from hip_helper import HIPRuntime, HIP_SUCCESS

            hip = HIPRuntime()
            # Try a basic allocation — should succeed via native passthrough
            err, ptr = hip.malloc_raw(1024 * 1024)  # 1 MiB
            print(f"err={err}")

            if err == HIP_SUCCESS:
                print("ALLOC_OK")
                hip.free(ptr)
            else:
                print(f"ALLOC_FAIL={err}")
        """,
            extra_env={"TF_SHM_FILE": "/tmp/nonexistent_cts_shm_file_12345"},
        )
        assert result.succeeded, f"Subprocess crashed with no SHM: {result.stderr}"
        # The limiter falls through to native HIP when SHM is missing.
        # Allocation must succeed (passthrough behavior).
        assert "ALLOC_OK" in result.stdout, (
            f"Expected native passthrough allocation to succeed, got: {result.stdout}"
        )


def test_multiple_alloc_sizes(cts):
    """Allocate various sizes and verify total SHM accounting matches sum of allocations."""
    result = cts.run_hip_test("""
        import os
        from hip_helper import HIPRuntime, HIP_SUCCESS
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]

        sizes = [4096, 65536, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]
        ptrs = []
        total_allocated = 0

        used_before = read_pod_memory_used(shm_path, 0)

        for size in sizes:
            err, ptr = hip.malloc_raw(size)
            if err != HIP_SUCCESS:
                print(f"FAIL: alloc of {size} failed with err={err}")
                break
            ptrs.append(ptr)
            total_allocated += size

        used_after = read_pod_memory_used(shm_path, 0)
        delta = used_after - used_before
        print(f"total_allocated={total_allocated}")
        print(f"shm_delta={delta}")

        # Clean up
        for ptr in ptrs:
            hip.free(ptr)

        used_final = read_pod_memory_used(shm_path, 0)
        print(f"used_final={used_final}")
        print(f"used_before={used_before}")

        if delta == total_allocated and used_final == used_before:
            print("PASS")
        else:
            print(f"FAIL: delta={delta} expected={total_allocated}, final={used_final} expected={used_before}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Multiple sizes test failed: {result.stdout}"
