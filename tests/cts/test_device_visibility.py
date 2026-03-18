"""
CTS Device Visibility Tests — TF_VISIBLE_DEVICES remapping and device-specific accounting.

Validates that the limiter correctly maps HIP device indices to SHM device entries
based on TF_VISIBLE_DEVICES. This is critical for multi-GPU systems where only a
subset of physical devices are allocated to a pod.

How device mapping works:
1. TF_VISIBLE_DEVICES contains comma-separated UUIDs (PCI BDF addresses)
2. The limiter matches these against hipDeviceGetPCIBusId for each HIP device
3. Matching devices get HIP_VISIBLE_DEVICES set so HIP only sees allocated GPUs
4. SHM device entries are indexed by the raw device index from gpu_idx_uuids
5. hipGetDevice() returns a virtual device ID (0-based within visible devices)
6. device_index_by_hip_device() maps virtual -> raw index via PCI bus ID lookup
"""

import pytest

from conftest import DEFAULT_TEST_UUID, MiB, GiB, parse_kv_output, requires_gpu, _discover_gpu_pci_bus_id
from shm_writer import DeviceSpec

pytestmark = requires_gpu


class TestVisibleDevicesRemap:
    """Verify the limiter correctly maps TF_VISIBLE_DEVICES UUIDs to device indices."""

    def test_visible_devices_remap(self, cts_factory):
        """Set TF_VISIBLE_DEVICES to a specific UUID and verify that allocations
        on device 0 (the only visible device) are tracked to the correct SHM entry.

        The SHM file has one device at index 0. TF_VISIBLE_DEVICES contains that
        device's UUID. The limiter should map HIP device 0 -> SHM device 0."""
        test_uuid = DEFAULT_TEST_UUID
        mem_limit = 512 * MiB

        fixture = cts_factory(
            devices=[
                DeviceSpec(uuid=test_uuid, mem_limit=mem_limit, device_idx=0),
            ],
        )

        alloc_size = 4 * MiB
        script = f"""\
import os
from hip_helper import HIPRuntime, HIP_SUCCESS
from shm_writer import read_pod_memory_used

hip = HIPRuntime()

# Verify we can see devices
count = hip.get_device_count()
print(f"DEVICE_COUNT={{count}}")

# Get the PCI bus ID of device 0 to confirm it matches our UUID
pci_id = hip.get_pci_bus_id(0)
print(f"PCI_BUS_ID={{pci_id}}")

# Allocate on device 0
hip.set_device(0)
err, ptr = hip.malloc_raw({alloc_size})
if err == HIP_SUCCESS and ptr != 0:
    print("ALLOC_OK")
    # Check spoofed memory info
    free_mem, total_mem = hip.mem_get_info()
    print(f"TOTAL_MEM={{total_mem}}")
    print(f"FREE_MEM={{free_mem}}")
    shm_used = read_pod_memory_used(os.environ["TF_SHM_FILE"], 0)
    print(f"SHM_USED={{shm_used}}")
else:
    print(f"ALLOC_FAIL={{err}}")
"""
        result = fixture.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "ALLOC_OK" in result.stdout, f"Allocation failed:\n{result.stdout}"

        # Parse output
        values = parse_kv_output(result.stdout)

        # Verify spoofed total matches our configured limit
        total_mem = values.get("TOTAL_MEM", 0)
        assert total_mem == mem_limit, (
            f"Spoofed total mem ({total_mem}) != configured limit ({mem_limit})"
        )

        # Verify SHM accounting for device 0 (reported by subprocess before atexit)
        shm_used = values.get("SHM_USED", 0)
        assert shm_used == alloc_size, (
            f"SHM device 0 pod_memory_used ({shm_used}) != alloc size ({alloc_size})"
        )

    def test_visible_devices_multi_device(self, cts_factory):
        """With two devices configured in SHM and TF_VISIBLE_DEVICES, allocations
        on each HIP device should be tracked to the correct SHM device entry.

        Note: This test requires a system with at least 2 AMD GPUs. If only 1 GPU
        is available, it will be skipped."""
        uuid_0 = _discover_gpu_pci_bus_id(0)
        uuid_1 = _discover_gpu_pci_bus_id(1)
        mem_limit = 256 * MiB

        fixture = cts_factory(
            devices=[
                DeviceSpec(uuid=uuid_0, mem_limit=mem_limit, device_idx=0),
                DeviceSpec(uuid=uuid_1, mem_limit=mem_limit, device_idx=1),
            ],
        )

        alloc_size_dev0 = 2 * MiB
        alloc_size_dev1 = 4 * MiB

        script = f"""\
import os
from hip_helper import HIPRuntime, HIP_SUCCESS
from shm_writer import read_pod_memory_used

hip = HIPRuntime()
count = hip.get_device_count()
print(f"DEVICE_COUNT={{count}}")

if count < 2:
    print("SKIP_NOT_ENOUGH_DEVICES")
    exit(0)

# Allocate on device 0
hip.set_device(0)
err0, ptr0 = hip.malloc_raw({alloc_size_dev0})
if err0 == HIP_SUCCESS and ptr0 != 0:
    print("DEV0_ALLOC_OK")
else:
    print(f"DEV0_ALLOC_FAIL={{err0}}")

# Allocate on device 1
hip.set_device(1)
err1, ptr1 = hip.malloc_raw({alloc_size_dev1})
if err1 == HIP_SUCCESS and ptr1 != 0:
    print("DEV1_ALLOC_OK")
else:
    print(f"DEV1_ALLOC_FAIL={{err1}}")

# Read SHM before exit (atexit handler will drain these)
shm_file = os.environ["TF_SHM_FILE"]
print(f"DEV0_SHM_USED={{read_pod_memory_used(shm_file, 0)}}")
print(f"DEV1_SHM_USED={{read_pod_memory_used(shm_file, 1)}}")
"""
        result = fixture.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"

        if "SKIP_NOT_ENOUGH_DEVICES" in result.stdout:
            pytest.skip("System has fewer than 2 AMD GPUs")

        assert "DEV0_ALLOC_OK" in result.stdout, f"Device 0 alloc failed:\n{result.stdout}"
        assert "DEV1_ALLOC_OK" in result.stdout, f"Device 1 alloc failed:\n{result.stdout}"

        # Each device's SHM entry should track its own allocations independently
        values = parse_kv_output(result.stdout)
        dev0_used = values.get("DEV0_SHM_USED", -1)
        dev1_used = values.get("DEV1_SHM_USED", -1)

        assert dev0_used == alloc_size_dev0, (
            f"Device 0 pod_memory_used ({dev0_used}) != expected ({alloc_size_dev0})"
        )
        assert dev1_used == alloc_size_dev1, (
            f"Device 1 pod_memory_used ({dev1_used}) != expected ({alloc_size_dev1})"
        )


class TestSingleDeviceAlloc:
    """With only 1 visible device, all allocations must be tracked to device 0."""

    def test_single_device_alloc(self, cts):
        """Using the default fixture (1 device), perform multiple allocations
        and verify they are all tracked to SHM device index 0."""
        sizes = [1 * MiB, 2 * MiB, 512 * 1024]
        total_expected = sum(sizes)
        sizes_str = repr(sizes)

        script = f"""\
import os
from hip_helper import HIPRuntime, HIP_SUCCESS
from shm_writer import read_pod_memory_used

hip = HIPRuntime()
count = hip.get_device_count()
print(f"DEVICE_COUNT={{count}}")

# All allocs go to device 0 (the only visible device)
hip.set_device(0)

sizes = {sizes_str}
pointers = []
for size in sizes:
    err, ptr = hip.malloc_raw(size)
    if err != HIP_SUCCESS or ptr == 0:
        print(f"ALLOC_FAIL={{err}} size={{size}}")
        exit(1)
    pointers.append(ptr)

print(f"ALL_ALLOCS_OK count={{len(pointers)}}")
shm_used = read_pod_memory_used(os.environ["TF_SHM_FILE"], 0)
print(f"SHM_USED={{shm_used}}")
"""
        result = cts.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "ALL_ALLOCS_OK" in result.stdout, f"Allocs failed:\n{result.stdout}"

        values = parse_kv_output(result.stdout)
        shm_used = values.get("SHM_USED", 0)
        assert shm_used == total_expected, (
            f"SHM device 0 pod_memory_used ({shm_used}) != expected ({total_expected})"
        )

    # NOTE: Basic info spoofing (hipMemGetInfo, hipDeviceTotalMem) is tested in
    # test_info_spoofing.py. This file focuses on device visibility/remapping.

    def test_single_device_free_decreases(self, cts):
        """After allocating on the single device, hipMemGetInfo free should decrease."""
        alloc_size = 4 * MiB
        script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS
hip = HIPRuntime()

# Get initial free memory
free_before, total = hip.mem_get_info()
print(f"FREE_BEFORE={{free_before}}")
print(f"TOTAL={{total}}")

# Allocate
err, ptr = hip.malloc_raw({alloc_size})
assert err == HIP_SUCCESS, f"Allocation failed: {{err}}"
print("ALLOC_OK")

# Get free memory after allocation
free_after, total_after = hip.mem_get_info()
print(f"FREE_AFTER={{free_after}}")

# Free and check again
hip.free(ptr)
free_final, _ = hip.mem_get_info()
print(f"FREE_FINAL={{free_final}}")
"""
        result = cts.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "ALLOC_OK" in result.stdout

        values = parse_kv_output(result.stdout)

        free_before = values["FREE_BEFORE"]
        free_after = values["FREE_AFTER"]
        free_final = values["FREE_FINAL"]

        # After alloc, free should decrease by alloc_size
        assert free_after == free_before - alloc_size, (
            f"Free mem after alloc ({free_after}) != before ({free_before}) - size ({alloc_size})"
        )

        # After free, free should return to original
        assert free_final == free_before, (
            f"Free mem after free ({free_final}) != original ({free_before})"
        )
