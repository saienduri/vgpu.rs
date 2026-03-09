"""
Free/Deallocation Tracking Tests — Validates that the hip-limiter correctly
tracks memory frees and decrements SHM pod_memory_used.

Key behaviors under test:
  - hipFree decrements pod_memory_used by the original allocation size
  - hipFree(NULL) is a no-op (no crash, no accounting change)
  - Double free of the same pointer is safe (second free is a no-op)
  - Freeing an untracked pointer does not affect accounting
  - hipHostFree and hipFreeAsync are tracked like hipFree
"""

import pytest

from conftest import MiB, GiB, requires_gpu
from shm_writer import DeviceSpec

pytestmark = requires_gpu


def test_free_decrements_usage(cts):
    """Allocate, verify SHM used increased, free, verify SHM used decreased."""
    result = cts.run_hip_test("""
        import os
        from hip_helper import HIPRuntime, HIP_SUCCESS
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]
        alloc_size = 32 * 1024 * 1024  # 32 MiB

        used_before = read_pod_memory_used(shm_path, 0)
        ptr = hip.malloc(alloc_size)
        used_after_alloc = read_pod_memory_used(shm_path, 0)
        hip.free(ptr)
        used_after_free = read_pod_memory_used(shm_path, 0)

        print(f"before={used_before}")
        print(f"after_alloc={used_after_alloc}")
        print(f"after_free={used_after_free}")

        delta_alloc = used_after_alloc - used_before
        delta_free = used_after_alloc - used_after_free

        if delta_alloc == alloc_size and delta_free == alloc_size:
            print("PASS")
        else:
            print(f"FAIL: delta_alloc={delta_alloc} delta_free={delta_free} expected={alloc_size}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Free tracking failed: {result.stdout}"


def test_free_null_no_crash(cts):
    """hipFree(NULL) should not crash and should not affect accounting."""
    result = cts.run_hip_test("""
        import os
        from hip_helper import HIPRuntime
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]

        used_before = read_pod_memory_used(shm_path, 0)

        # hipFree(NULL) — the Rust hook checks for null and skips record_free
        err = hip.free_raw(0)
        print(f"err={err}")

        used_after = read_pod_memory_used(shm_path, 0)
        print(f"before={used_before}")
        print(f"after={used_after}")

        if used_before == used_after:
            print("PASS")
        else:
            print(f"FAIL: usage changed from {used_before} to {used_after}")
    """)
    assert result.succeeded, f"Subprocess failed (crash?): {result.stderr}"
    assert "PASS" in result.stdout, f"Free null test failed: {result.stdout}"


def test_double_free_safety(cts):
    """Freeing the same pointer twice should be safe — the second free is a no-op
    in the limiter's accounting (record_free returns false for unknown pointers).

    Note: The underlying HIP runtime may or may not error on double free, but
    the limiter should not crash or corrupt accounting.
    """
    result = cts.run_hip_test("""
        import os
        from hip_helper import HIPRuntime, HIP_SUCCESS
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]
        alloc_size = 16 * 1024 * 1024  # 16 MiB

        ptr = hip.malloc(alloc_size)
        used_after_alloc = read_pod_memory_used(shm_path, 0)

        # First free — should decrement
        hip.free(ptr)
        used_after_first_free = read_pod_memory_used(shm_path, 0)

        # Second free — limiter should no-op (ptr already removed from tracker)
        # The native hipFree may return an error, but we use free_raw to avoid exceptions
        err = hip.free_raw(ptr)
        used_after_second_free = read_pod_memory_used(shm_path, 0)

        print(f"after_alloc={used_after_alloc}")
        print(f"after_first_free={used_after_first_free}")
        print(f"after_second_free={used_after_second_free}")
        print(f"second_free_err={err}")

        # Key invariant: second free should NOT decrement further
        if used_after_first_free == used_after_second_free:
            print("PASS")
        else:
            print(f"FAIL: second free changed usage from {used_after_first_free} to {used_after_second_free}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Double free test failed: {result.stdout}"


def test_free_untracked_ptr(cts):
    """Freeing a pointer not allocated through the hooked APIs should not
    affect the limiter's accounting. The limiter's record_free returns false
    for unknown pointers.
    """
    result = cts.run_hip_test("""
        import os
        from hip_helper import HIPRuntime
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]

        # Allocate something to establish baseline
        ptr = hip.malloc(8 * 1024 * 1024)  # 8 MiB
        used_with_alloc = read_pod_memory_used(shm_path, 0)

        # Free an arbitrary address (0xDEADBEEF) — this is an untracked pointer.
        # The limiter should not find it in the tracker and should not change accounting.
        # The native hipFree will likely error, but the limiter's accounting stays intact.
        err = hip.free_raw(0xDEADBEEF)
        used_after_bad_free = read_pod_memory_used(shm_path, 0)

        print(f"used_with_alloc={used_with_alloc}")
        print(f"used_after_bad_free={used_after_bad_free}")
        print(f"bad_free_err={err}")

        if used_with_alloc == used_after_bad_free:
            print("PASS")
        else:
            print(f"FAIL: bad free changed usage from {used_with_alloc} to {used_after_bad_free}")

        # Clean up the real allocation
        hip.free(ptr)
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Untracked free test failed: {result.stdout}"


def test_host_free_tracking(cts):
    """hipHostMalloc + hipHostFree pair should be tracked in SHM accounting."""
    result = cts.run_hip_test("""
        import os
        from hip_helper import HIPRuntime
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]
        alloc_size = 16 * 1024 * 1024  # 16 MiB

        used_before = read_pod_memory_used(shm_path, 0)
        ptr = hip.host_malloc(alloc_size, 0)
        used_after_alloc = read_pod_memory_used(shm_path, 0)
        hip.host_free(ptr)
        used_after_free = read_pod_memory_used(shm_path, 0)

        print(f"before={used_before}")
        print(f"after_alloc={used_after_alloc}")
        print(f"after_free={used_after_free}")

        delta_alloc = used_after_alloc - used_before
        delta_free = used_after_alloc - used_after_free

        if delta_alloc == alloc_size and delta_free == alloc_size:
            print("PASS")
        else:
            print(f"FAIL: delta_alloc={delta_alloc} delta_free={delta_free} expected={alloc_size}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Host free tracking failed: {result.stdout}"


def test_async_free_tracking(cts):
    """hipMallocAsync + hipFreeAsync pair should be tracked in SHM accounting."""
    result = cts.run_hip_test("""
        import os
        from hip_helper import HIPRuntime
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]
        alloc_size = 16 * 1024 * 1024  # 16 MiB

        used_before = read_pod_memory_used(shm_path, 0)
        ptr = hip.malloc_async(alloc_size, 0)
        used_after_alloc = read_pod_memory_used(shm_path, 0)

        hip.free_async(ptr, 0)
        hip.device_synchronize()
        used_after_free = read_pod_memory_used(shm_path, 0)

        print(f"before={used_before}")
        print(f"after_alloc={used_after_alloc}")
        print(f"after_free={used_after_free}")

        delta_alloc = used_after_alloc - used_before
        delta_free = used_after_alloc - used_after_free

        if delta_alloc == alloc_size and delta_free == alloc_size:
            print("PASS")
        else:
            print(f"FAIL: delta_alloc={delta_alloc} delta_free={delta_free} expected={alloc_size}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Async free tracking failed: {result.stdout}"
