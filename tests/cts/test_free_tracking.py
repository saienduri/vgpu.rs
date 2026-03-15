"""
Free/Deallocation Tracking Tests — Validates that the hip-limiter correctly
tracks memory frees and decrements SHM pod_memory_used.

Key behaviors under test:
  - hipFree decrements pod_memory_used by the original allocation size
  - hipFree(NULL) is a no-op (no crash, no accounting change)
  - Double free of the same pointer is safe (second free is a no-op)
  - Freeing an untracked pointer does not affect accounting
  - hipHostFree, hipFreeHost, and hipFreeAsync are tracked like hipFree
  - Cross-pairing (e.g., hipMallocHost alloc + hipFreeHost free) works correctly
"""

import pytest

from conftest import MiB, GiB, requires_gpu
from shm_writer import DeviceSpec

pytestmark = requires_gpu


# --- Alloc/free pair tracking (parametrized) ---

# Each entry: (test_id, alloc_call, free_call, post_free_call)
# alloc_call and free_call are Python expressions using `hip` and `alloc_size`.
# post_free_call is an optional expression to run after free (e.g., device_synchronize).
_FREE_TRACKING_PAIRS = [
    ("hipMalloc+hipFree", "hip.malloc(alloc_size)", "hip.free(ptr)", None),
    ("hipHostMalloc+hipHostFree", "hip.host_malloc(alloc_size, 0)", "hip.host_free(ptr)", None),
    ("hipHostMalloc+hipFreeHost", "hip.host_malloc(alloc_size, 0)", "hip.free_host(ptr)", None),
    ("hipHostAlloc+hipFreeHost", "hip.host_alloc(alloc_size, 0)", "hip.free_host(ptr)", None),
    ("hipMallocHost+hipFreeHost", "hip.malloc_host(alloc_size)", "hip.free_host(ptr)", None),
    ("hipMemAllocHost+hipFreeHost", "hip.mem_alloc_host(alloc_size)", "hip.free_host(ptr)", None),
    ("hipMemAllocHost+hipHostFree", "hip.mem_alloc_host(alloc_size)", "hip.host_free(ptr)", None),
    ("hipMallocAsync+hipFreeAsync", "hip.malloc_async(alloc_size, 0)", "hip.free_async(ptr, 0)", "hip.device_synchronize()"),
]


@pytest.mark.parametrize(
    "alloc_call,free_call,post_free",
    [(a, f, p) for _, a, f, p in _FREE_TRACKING_PAIRS],
    ids=[tid for tid, _, _, _ in _FREE_TRACKING_PAIRS],
)
def test_free_pair_tracking(cts, alloc_call, free_call, post_free):
    """Alloc/free pairs should be tracked symmetrically in SHM accounting."""
    post_free_line = f"\n        {post_free}" if post_free else ""
    result = cts.run_hip_test(f"""
        import os
        from hip_helper import HIPRuntime
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]
        alloc_size = 16 * 1024 * 1024  # 16 MiB

        used_before = read_pod_memory_used(shm_path, 0)
        ptr = {alloc_call}
        used_after_alloc = read_pod_memory_used(shm_path, 0)
        {free_call}{post_free_line}
        used_after_free = read_pod_memory_used(shm_path, 0)

        print(f"before={{used_before}}")
        print(f"after_alloc={{used_after_alloc}}")
        print(f"after_free={{used_after_free}}")

        delta_alloc = used_after_alloc - used_before
        delta_free = used_after_alloc - used_after_free

        if delta_alloc == alloc_size and delta_free == alloc_size:
            print("PASS")
        else:
            print(f"FAIL: delta_alloc={{delta_alloc}} delta_free={{delta_free}} expected={{alloc_size}}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Free tracking failed: {result.stdout}"


# --- Pitched alloc free tracking ---
#
# Pitched allocs return (ptr, pitch) so they don't fit the parametrized template
# above (which expects `ptr = <expr>`). The tracked size is pitch*height, not
# width*height, so the delta check also differs from the simple alloc pairs.

_PITCHED_FREE_TRACKING_PAIRS = [
    ("hipMallocPitch+hipFree", "hip.malloc_pitch(width, height)", "hip.free(ptr)"),
    ("hipMemAllocPitch+hipFree", "hip.mem_alloc_pitch(width, height)", "hip.free(ptr)"),
]


@pytest.mark.parametrize(
    "alloc_call,free_call",
    [(a, f) for _, a, f in _PITCHED_FREE_TRACKING_PAIRS],
    ids=[tid for tid, _, _ in _PITCHED_FREE_TRACKING_PAIRS],
)
def test_pitched_free_returns_to_zero(cts, alloc_call, free_call):
    """Pitched alloc + free should return pod_memory_used to zero."""
    width, height = 1024, 512
    result = cts.run_hip_test(f"""
        import os
        from hip_helper import HIPRuntime
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]
        width, height = {width}, {height}

        used_before = read_pod_memory_used(shm_path, 0)
        ptr, pitch = {alloc_call}
        used_after_alloc = read_pod_memory_used(shm_path, 0)
        {free_call}
        used_after_free = read_pod_memory_used(shm_path, 0)

        delta_alloc = used_after_alloc - used_before
        expected = pitch * height

        print(f"delta_alloc={{delta_alloc}}")
        print(f"expected={{expected}}")
        print(f"after_free={{used_after_free}}")

        if delta_alloc == expected and used_after_free == used_before:
            print("PASS")
        else:
            print(f"FAIL: delta_alloc={{delta_alloc}} expected={{expected}} after_free={{used_after_free}} before={{used_before}}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Pitched free tracking failed: {result.stdout}"


# --- Handle-based free tracking (hipMemCreate/hipMemRelease) ---
#
# hipMemCreate uses opaque handles rather than device pointers, so it doesn't
# fit the parametrized template above. The alloc size must be granularity-aligned.

def test_mem_create_release_returns_to_zero(cts):
    """hipMemRelease should return pod_memory_used to zero after releasing a hipMemCreate handle."""
    result = cts.run_hip_test("""
        import os
        from hip_helper import HIPRuntime
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]

        granularity = hip.get_allocation_granularity(0)
        alloc_size = ((16 * 1024 * 1024 + granularity - 1) // granularity) * granularity

        used_before = read_pod_memory_used(shm_path, 0)
        handle = hip.mem_create(alloc_size, 0)
        hip.mem_release(handle)
        used_after_release = read_pod_memory_used(shm_path, 0)

        print(f"before={used_before}")
        print(f"after_release={used_after_release}")

        if used_after_release == used_before:
            print("PASS")
        else:
            print(f"FAIL: expected {used_before}, got {used_after_release}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"hipMemRelease tracking failed: {result.stdout}"


# --- Edge cases (not parametrizable — unique logic per test) ---

def test_free_null_no_crash(cts):
    """hipFree(NULL) should not crash and should not affect accounting."""
    result = cts.run_hip_test("""
        import os
        from hip_helper import HIPRuntime
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]

        used_before = read_pod_memory_used(shm_path, 0)

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


def test_free_host_null_no_crash(cts):
    """hipFreeHost(NULL) should not crash and should not affect accounting."""
    result = cts.run_hip_test("""
        import os
        from hip_helper import HIPRuntime
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]

        used_before = read_pod_memory_used(shm_path, 0)

        err = hip.free_host_raw(0)
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
    assert "PASS" in result.stdout, f"hipFreeHost null test failed: {result.stdout}"


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
