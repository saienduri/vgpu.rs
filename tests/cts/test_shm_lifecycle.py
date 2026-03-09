"""
CTS SHM Lifecycle Tests — Heartbeat staleness, missing SHM, and accounting accuracy.

Validates the limiter's behavior under various SHM lifecycle scenarios:
- Stale heartbeat: limiter warns but continues enforcement (no passthrough)
- Missing SHM: limiter fails to init, all HIP calls pass through unintercepted
- Accounting accuracy: after known allocations, SHM pod_memory_used matches exactly
"""

import os
import shutil
import tempfile

import pytest

from conftest import DEFAULT_TEST_UUID, MiB, GiB, parse_kv_output, requires_gpu
from shm_writer import (
    DeviceSpec,
    update_heartbeat,
)

pytestmark = requires_gpu


class TestStaleHeartbeatEnforcement:
    """Stale heartbeat logs a warning but enforcement continues normally.

    The limiter always enforces memory limits regardless of heartbeat state.
    The heartbeat is a management-plane signal (hypervisor liveness), not a
    data-plane one — the SHM limits are immutable after pod creation, so
    there's no reason to stop enforcing when the hypervisor is temporarily down.
    """

    def test_stale_heartbeat_still_enforces(self, cts_no_heartbeat):
        """Set heartbeat to epoch 0 (far in the past), then allocate.
        The limiter should warn about stale heartbeat but still enforce
        limits and track the allocation in SHM."""
        cts = cts_no_heartbeat

        # Set heartbeat to epoch 0 — definitely stale (> 2s threshold)
        update_heartbeat(cts.shm_path, timestamp=0)

        alloc_size = 4 * MiB
        script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS
hip = HIPRuntime()
err, ptr = hip.malloc_raw({alloc_size})
if err == HIP_SUCCESS and ptr != 0:
    print("ALLOC_OK")
    # Free to clean up
    hip.free(ptr)
    print("FREE_OK")
else:
    print(f"ALLOC_RESULT={{err}}")
"""
        result = cts.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"

        # Allocation should succeed (within limit) even with stale heartbeat.
        assert "ALLOC_OK" in result.stdout, (
            f"Expected allocation to succeed with stale heartbeat, "
            f"got:\n{result.stdout}"
        )

        # Verify the stale heartbeat warning was logged.
        assert "stale heartbeat" in result.output.lower(), (
            f"Expected 'stale heartbeat' warning in output but got:\n{result.output}"
        )

    def test_stale_heartbeat_tracks_in_shm(self, cts_no_heartbeat):
        """With a stale heartbeat, the limiter still enforces and tracks
        allocations in SHM (pod_memory_used reflects the allocation)."""
        cts = cts_no_heartbeat
        update_heartbeat(cts.shm_path, timestamp=0)

        alloc_size = 4 * MiB
        script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS
hip = HIPRuntime()
err, ptr = hip.malloc_raw({alloc_size})
if err == HIP_SUCCESS:
    print("ALLOC_OK")
else:
    print(f"ALLOC_FAIL={{err}}")
"""
        result = cts.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "ALLOC_OK" in result.stdout

        # Stale heartbeat no longer causes passthrough — allocation is tracked in SHM.
        shm_used = cts.read_pod_memory_used(device_idx=0)
        assert shm_used == alloc_size, (
            f"SHM pod_memory_used ({shm_used}) should be {alloc_size} — "
            f"stale heartbeat no longer bypasses enforcement"
        )

    def test_stale_heartbeat_info_spoofing_works(self, cts_no_heartbeat):
        """With a stale heartbeat, hipMemGetInfo and hipDeviceTotalMem should
        still return spoofed values (not errors). The heartbeat is a management-
        plane signal; info spoofing depends only on the immutable SHM limits."""
        cts = cts_no_heartbeat
        update_heartbeat(cts.shm_path, timestamp=0)

        script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS
hip = HIPRuntime()

free_mem, total_mem = hip.mem_get_info()
print(f"TOTAL={{total_mem}}")
print(f"FREE={{free_mem}}")

device_total = hip.device_total_mem(0)
print(f"DEVICE_TOTAL={{device_total}}")

expected = {1 * GiB}
if total_mem == expected and device_total == expected:
    print("PASS")
else:
    print(f"FAIL: expected {{expected}}, total={{total_mem}}, device_total={{device_total}}")
"""
        result = cts.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "PASS" in result.stdout, (
            f"Info spoofing should work with stale heartbeat:\n{result.stdout}"
        )


class TestMissingSHMPassthrough:
    """When TF_SHM_FILE points to a nonexistent file, the limiter cannot initialize.
    GLOBAL_LIMITER will not be set, hooks will not be installed, and all HIP calls
    pass through to the native implementation."""

    def test_missing_shm_passthrough(self, cts_factory):
        """Point TF_SHM_FILE to a nonexistent path. HIP calls should work normally
        without enforcement."""
        # Create a fixture pointing to a nonexistent SHM file.
        # We use cts_factory with extra_env to override TF_SHM_FILE after setup.
        # The approach: create a normal fixture, then override the SHM path in env.
        nonexistent_dir = tempfile.mkdtemp(prefix="cts_missing_")
        nonexistent_shm = os.path.join(nonexistent_dir, "subdir", "shm")
        # Intentionally do NOT create the file — it should not exist

        fixture = cts_factory(
            devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=1 * GiB, device_idx=0)]
        )

        script = """\
from hip_helper import HIPRuntime, HIP_SUCCESS
hip = HIPRuntime()

# Basic allocation should work (passthrough, no limiter)
err, ptr = hip.malloc_raw(1024 * 1024)  # 1 MiB
if err == HIP_SUCCESS and ptr != 0:
    print("ALLOC_OK")
    hip.free(ptr)
    print("FREE_OK")
else:
    print(f"ALLOC_RESULT={err}")

# Memory info should return real hardware values (not spoofed)
free_mem, total_mem = hip.mem_get_info()
print(f"TOTAL_MEM={total_mem}")
print(f"FREE_MEM={free_mem}")
"""
        # Override TF_SHM_FILE to point to nonexistent path
        result = fixture.run_hip_test(
            script,
            extra_env={"TF_SHM_FILE": nonexistent_shm},
        )
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "ALLOC_OK" in result.stdout, (
            f"Allocation should pass through when SHM is missing:\n{result.stdout}"
        )
        assert "FREE_OK" in result.stdout

        # Clean up the temp dir (may contain pytest artifacts)
        shutil.rmtree(nonexistent_dir, ignore_errors=True)

    def test_no_shm_env_passthrough(self, cts_factory):
        """When neither TF_SHM_FILE nor hypervisor env is set, limiter should not
        init, and HIP calls pass through."""
        fixture = cts_factory(
            devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=1 * GiB, device_idx=0)]
        )

        script = """\
from hip_helper import HIPRuntime, HIP_SUCCESS
hip = HIPRuntime()
err, ptr = hip.malloc_raw(1024 * 1024)
if err == HIP_SUCCESS and ptr != 0:
    print("ALLOC_OK")
    hip.free(ptr)
else:
    print(f"ALLOC_RESULT={err}")
"""
        # Remove TF_SHM_FILE and TF_VISIBLE_DEVICES — no way for limiter to init
        result = fixture.run_hip_test(
            script,
            extra_env={
                "TF_SHM_FILE": None,
                "TF_VISIBLE_DEVICES": None,
            },
        )
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "ALLOC_OK" in result.stdout, (
            f"Allocation should pass through with no SHM config:\n{result.stdout}"
        )


class TestUnlimitedDevicePassthrough:
    """When up_limit >= 100, the limiter considers all devices unlimited.

    Combined with TF_SKIP_HOOKS_IF_NO_LIMIT=true, hooks are not installed and
    allocations pass through to the native HIP allocator without enforcement.
    """

    def test_unlimited_device_allows_over_limit(self, cts_factory):
        """With up_limit=100 and TF_SKIP_HOOKS_IF_NO_LIMIT=true, allocations
        exceeding mem_limit should succeed because hooks are bypassed entirely."""
        mem_limit = 4 * MiB
        fixture = cts_factory(
            devices=[
                DeviceSpec(
                    uuid=DEFAULT_TEST_UUID,
                    mem_limit=mem_limit,
                    up_limit=100,
                    device_idx=0,
                )
            ],
            extra_env={"TF_SKIP_HOOKS_IF_NO_LIMIT": "true"},
        )

        alloc_size = 8 * MiB  # 2x the mem_limit
        script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS
hip = HIPRuntime()
err, ptr = hip.malloc_raw({alloc_size})
if err == HIP_SUCCESS and ptr != 0:
    print("ALLOC_OK")
    hip.free(ptr)
    print("FREE_OK")
else:
    print(f"ALLOC_FAIL={{err}}")
"""
        result = fixture.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "ALLOC_OK" in result.stdout, (
            f"Expected allocation to succeed with up_limit=100 (hooks bypassed), "
            f"got:\n{result.stdout}"
        )

    def test_skip_hooks_default_still_enforces(self, cts_factory):
        """Without TF_SKIP_HOOKS_IF_NO_LIMIT set (default), hooks are installed
        even when up_limit=100. Allocations exceeding mem_limit should be denied."""
        mem_limit = 4 * MiB
        fixture = cts_factory(
            devices=[
                DeviceSpec(
                    uuid=DEFAULT_TEST_UUID,
                    mem_limit=mem_limit,
                    up_limit=100,
                    device_idx=0,
                )
            ],
            # TF_SKIP_HOOKS_IF_NO_LIMIT not set — hooks remain active
        )

        alloc_size = 8 * MiB  # 2x the mem_limit
        script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY
hip = HIPRuntime()
err, ptr = hip.malloc_raw({alloc_size})
print(f"ERR={{err}}")
if err == HIP_ERROR_OUT_OF_MEMORY:
    print("OOM_OK")
elif err == HIP_SUCCESS:
    print("ALLOC_OK")
    hip.free(ptr)
"""
        result = fixture.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "OOM_OK" in result.stdout, (
            f"Expected OOM when TF_SKIP_HOOKS_IF_NO_LIMIT is not set "
            f"(hooks should still enforce), got:\n{result.stdout}"
        )

    def test_skip_hooks_limited_device_still_enforces(self, cts_factory):
        """With TF_SKIP_HOOKS_IF_NO_LIMIT=true but up_limit < 100, hooks should
        still be installed and enforcement should work."""
        mem_limit = 4 * MiB
        fixture = cts_factory(
            devices=[
                DeviceSpec(
                    uuid=DEFAULT_TEST_UUID,
                    mem_limit=mem_limit,
                    up_limit=80,  # Not unlimited
                    device_idx=0,
                )
            ],
            extra_env={"TF_SKIP_HOOKS_IF_NO_LIMIT": "true"},
        )

        alloc_size = 8 * MiB  # 2x the mem_limit
        script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY
hip = HIPRuntime()
err, ptr = hip.malloc_raw({alloc_size})
print(f"ERR={{err}}")
if err == HIP_ERROR_OUT_OF_MEMORY:
    print("OOM_OK")
elif err == HIP_SUCCESS:
    print("ALLOC_OK")
    hip.free(ptr)
"""
        result = fixture.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "OOM_OK" in result.stdout, (
            f"Expected OOM when up_limit=80 (hooks should enforce even with "
            f"TF_SKIP_HOOKS_IF_NO_LIMIT=true), got:\n{result.stdout}"
        )


class TestSHMAccountingMatchesAllocs:
    """After N allocations of known sizes, SHM pod_memory_used must exactly equal
    the sum of those sizes."""

    def test_shm_accounting_matches_allocs(self, cts):
        """Perform several allocations of known sizes and verify SHM matches."""
        sizes = [1 * MiB, 2 * MiB, 4 * MiB, 512 * 1024, 256 * 1024]
        total_expected = sum(sizes)
        sizes_str = repr(sizes)

        script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS
hip = HIPRuntime()

sizes = {sizes_str}
pointers = []
for size in sizes:
    err, ptr = hip.malloc_raw(size)
    if err != HIP_SUCCESS or ptr == 0:
        print(f"ALLOC_FAIL={{err}} size={{size}}")
        # Clean up any successful allocs
        for p in pointers:
            hip.free(p)
        exit(1)
    pointers.append(ptr)

print(f"ALL_ALLOCS_OK count={{len(pointers)}}")
# Leave allocations live — do not free
"""
        result = cts.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "ALL_ALLOCS_OK" in result.stdout, f"Not all allocs succeeded:\n{result.stdout}"

        shm_used = cts.read_pod_memory_used(device_idx=0)
        assert shm_used == total_expected, (
            f"SHM pod_memory_used ({shm_used}) != expected sum of alloc sizes ({total_expected})"
        )

    def test_shm_accounting_after_partial_free(self, cts):
        """Allocate N buffers, free some, verify SHM tracks only live allocs."""
        alloc_size = 1 * MiB
        num_allocs = 6
        num_frees = 4  # Free first 4, keep last 2

        script = f"""\
import json
from hip_helper import HIPRuntime, HIP_SUCCESS

hip = HIPRuntime()
alloc_size = {alloc_size}
num_allocs = {num_allocs}
num_frees = {num_frees}

pointers = []
for i in range(num_allocs):
    err, ptr = hip.malloc_raw(alloc_size)
    if err != HIP_SUCCESS or ptr == 0:
        print(f"ALLOC_FAIL={{err}}")
        exit(1)
    pointers.append(ptr)

# Free the first num_frees pointers
for i in range(num_frees):
    hip.free(pointers[i])

live_count = num_allocs - num_frees
print(f"LIVE_COUNT={{live_count}}")
print(f"LIVE_BYTES={{live_count * alloc_size}}")
"""
        result = cts.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"

        values = parse_kv_output(result.stdout)
        expected_live_bytes = values.get("LIVE_BYTES", 0)
        assert expected_live_bytes > 0

        shm_used = cts.read_pod_memory_used(device_idx=0)
        assert shm_used == expected_live_bytes, (
            f"SHM pod_memory_used ({shm_used}) != expected live bytes ({expected_live_bytes})"
        )

    def test_shm_returns_to_zero_after_all_freed(self, cts):
        """Allocate and free everything; SHM should return to 0."""
        num_allocs = 10
        alloc_size = 512 * 1024  # 512 KiB each

        script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS

hip = HIPRuntime()
pointers = []
for _ in range({num_allocs}):
    err, ptr = hip.malloc_raw({alloc_size})
    if err != HIP_SUCCESS or ptr == 0:
        print(f"ALLOC_FAIL={{err}}")
        exit(1)
    pointers.append(ptr)

for ptr in pointers:
    hip.free(ptr)

print("ALL_FREED")
"""
        result = cts.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "ALL_FREED" in result.stdout

        shm_used = cts.read_pod_memory_used(device_idx=0)
        assert shm_used == 0, (
            f"SHM pod_memory_used ({shm_used}) should be 0 after all frees"
        )


class TestHooksDisabledPassthrough:
    """When ENABLE_HIP_HOOKS=false, the limiter .so is loaded but hooks are not
    installed. All HIP calls pass through to the native implementation, SHM is
    not written, and allocations beyond the configured limit succeed."""

    def test_hooks_disabled_alloc_exceeds_limit(self, cts_no_hooks):
        """With hooks disabled, allocations exceeding the SHM limit should
        succeed because no enforcement is active."""
        alloc_size = 2 * GiB  # 2x the default 1 GiB limit
        script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS
hip = HIPRuntime()
err, ptr = hip.malloc_raw({alloc_size})
if err == HIP_SUCCESS and ptr != 0:
    print("ALLOC_OK")
    hip.free(ptr)
    print("FREE_OK")
else:
    print(f"ALLOC_FAIL={{err}}")
"""
        result = cts_no_hooks.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "ALLOC_OK" in result.stdout, (
            f"Expected allocation to pass through with hooks disabled, "
            f"got:\n{result.stdout}"
        )

    def test_hooks_disabled_shm_unchanged(self, cts_no_hooks):
        """With hooks disabled, SHM pod_memory_used should remain 0
        regardless of allocations."""
        alloc_size = 64 * MiB
        script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS
hip = HIPRuntime()
err, ptr = hip.malloc_raw({alloc_size})
if err == HIP_SUCCESS:
    print("ALLOC_OK")
else:
    print(f"ALLOC_FAIL={{err}}")
"""
        result = cts_no_hooks.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "ALLOC_OK" in result.stdout

        shm_used = cts_no_hooks.read_pod_memory_used(device_idx=0)
        assert shm_used == 0, (
            f"SHM pod_memory_used ({shm_used}) should be 0 when hooks are disabled"
        )

    def test_hooks_disabled_info_not_spoofed(self, cts_no_hooks):
        """With hooks disabled, hipMemGetInfo should return real hardware values,
        not the SHM-configured limit."""
        script = f"""\
from hip_helper import HIPRuntime
hip = HIPRuntime()
free_mem, total_mem = hip.mem_get_info()
print(f"TOTAL={{total_mem}}")
print(f"FREE={{free_mem}}")
configured_limit = {1 * GiB}
if total_mem != configured_limit:
    print("NOT_SPOOFED")
else:
    print("POSSIBLY_SPOOFED")
"""
        result = cts_no_hooks.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "NOT_SPOOFED" in result.stdout, (
            f"hipMemGetInfo should return real GPU total when hooks are disabled, "
            f"got:\n{result.stdout}"
        )
