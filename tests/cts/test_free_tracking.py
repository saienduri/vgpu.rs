"""
Free/Deallocation Tracking Tests — Validates that the hip-limiter correctly
tracks memory frees and decrements SHM pod_memory_used.

Key behaviors under test:
  - hipFree decrements pod_memory_used by the original allocation size
  - hipFree(NULL) is a no-op (no crash, no accounting change)
  - Double free of the same pointer is safe (second free is a no-op)
  - Freeing an untracked pointer does not affect accounting
  - hipFreeAsync is tracked like hipFree
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


# --- Array free tracking (parametrized) ---
#
# HIP array APIs (hipMallocArray etc.) return hipErrorNotSupported (801) on some
# GPUs (e.g., MI325X/gfx942). Tests skip gracefully when the platform lacks support.

_ARRAY_FREE_TRACKING_CASES = [
    ("hipMallocArray+hipFreeArray",
     "hip.malloc_array(width=256, height=256, desc_x=32)",
     "hip.free_array(arr)",
     "4 * 256 * 256"),
    ("hipArrayCreate+hipArrayDestroy",
     "hip.array_create(width=256, height=256, fmt=HIP_AD_FORMAT_FLOAT, num_channels=1)",
     "hip.array_destroy(arr)",
     "4 * 1 * 256 * 256"),
    ("hipMallocArray+hipArrayDestroy",
     "hip.malloc_array(width=256, height=256, desc_x=32)",
     "hip.array_destroy(arr)",
     "4 * 256 * 256"),
    ("hipMalloc3DArray+hipFreeArray",
     "hip.malloc_3d_array(width=64, height=64, depth=4, desc_x=32)",
     "hip.free_array(arr)",
     "4 * 64 * 64 * 4"),
    ("hipArray3DCreate+hipArrayDestroy",
     "hip.array_3d_create(width=64, height=64, depth=4, fmt=HIP_AD_FORMAT_FLOAT, num_channels=1)",
     "hip.array_destroy(arr)",
     "4 * 1 * 64 * 64 * 4"),
    # --- Mipmapped array pairs ---
    # Mipmapped arrays account for the full mip chain (sum of all levels).
    # Like regular array APIs, these return hipErrorNotSupported (801) on MI325X/gfx942.
    ("hipMallocMipmappedArray+hipFreeMipmappedArray",
     "hip.malloc_mipmapped_array(width=256, height=256, num_levels=1, desc_x=32)",
     "hip.free_mipmapped_array(arr)",
     "4 * 256 * 256"),
    ("hipMipmappedArrayCreate+hipMipmappedArrayDestroy",
     "hip.mipmapped_array_create(width=256, height=256, num_levels=1, fmt=HIP_AD_FORMAT_FLOAT, num_channels=1)",
     "hip.mipmapped_array_destroy(arr)",
     "4 * 1 * 256 * 256"),
    ("hipMallocMipmappedArray+hipMipmappedArrayDestroy",
     "hip.malloc_mipmapped_array(width=256, height=256, num_levels=1, desc_x=32)",
     "hip.mipmapped_array_destroy(arr)",
     "4 * 256 * 256"),
    # Multi-level: 128x128, 3 levels => 4*(128*128 + 64*64 + 32*32) = 4*(16384+4096+1024) = 4*21504 = 86016
    ("hipMallocMipmappedArray+hipFreeMipmappedArray[3levels]",
     "hip.malloc_mipmapped_array(width=128, height=128, num_levels=3, desc_x=32)",
     "hip.free_mipmapped_array(arr)",
     "4 * (128*128 + 64*64 + 32*32)"),
    ("hipMipmappedArrayCreate+hipMipmappedArrayDestroy[3levels]",
     "hip.mipmapped_array_create(width=128, height=128, num_levels=3, fmt=HIP_AD_FORMAT_FLOAT, num_channels=1)",
     "hip.mipmapped_array_destroy(arr)",
     "4 * 1 * (128*128 + 64*64 + 32*32)"),
]


@pytest.mark.parametrize(
    "alloc_call,free_call,expected_expr",
    [(a, f, e) for _, a, f, e in _ARRAY_FREE_TRACKING_CASES],
    ids=[tid for tid, _, _, _ in _ARRAY_FREE_TRACKING_CASES],
)
def test_array_free_tracking(cts, alloc_call, free_call, expected_expr):
    """Array and mipmapped array alloc/free pairs should return pod_memory_used to zero."""
    result = cts.run_hip_test(f"""\
        import os
        from hip_helper import HIPRuntime, HIPError, HIP_AD_FORMAT_FLOAT, HIP_ERROR_NOT_SUPPORTED
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]

        used_before = read_pod_memory_used(shm_path, 0)
        try:
            arr = {alloc_call}
        except HIPError as e:
            if e.error_code == HIP_ERROR_NOT_SUPPORTED:
                print("NOT_SUPPORTED")
                raise SystemExit(0)
            raise
        used_after_alloc = read_pod_memory_used(shm_path, 0)
        {free_call}
        used_after_free = read_pod_memory_used(shm_path, 0)

        expected = {expected_expr}

        print(f"delta_alloc={{used_after_alloc - used_before}}")
        print(f"expected={{expected}}")
        print(f"after_free={{used_after_free}}")

        if used_after_alloc - used_before == expected and used_after_free == used_before:
            print("PASS")
        else:
            print(f"FAIL: delta={{used_after_alloc - used_before}} expected={{expected}} after_free={{used_after_free}} before={{used_before}}")
    """)
    if "NOT_SUPPORTED" in result.stdout:
        pytest.skip("Array/mipmapped array API not supported on this GPU")
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Array free tracking failed: {result.stdout}"
