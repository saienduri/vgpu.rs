"""
Info Spoofing Tests — Validates that hipMemGetInfo and hipDeviceTotalMem return
spoofed values reflecting the SHM-configured memory limit, not the real GPU memory.

Key behaviors under test:
  - hipMemGetInfo reports total == mem_limit from SHM
  - hipMemGetInfo reports free == mem_limit - pod_memory_used
  - hipDeviceTotalMem reports mem_limit from SHM
  - Both APIs return consistent values
"""

import pytest

from conftest import DEFAULT_TEST_UUID, MiB, GiB, requires_gpu, _discover_gpu_pci_bus_id, parse_kv_output
from shm_writer import DeviceSpec

pytestmark = requires_gpu


def test_mem_get_info_reports_limit(cts):
    """hipMemGetInfo should report total == the configured SHM mem_limit (1 GiB)."""
    result = cts.run_hip_test("""
        from hip_helper import HIPRuntime

        hip = HIPRuntime()
        free_mem, total_mem = hip.mem_get_info()

        print(f"free={free_mem}")
        print(f"total={total_mem}")

        expected_limit = 1024 * 1024 * 1024  # 1 GiB
        if total_mem == expected_limit:
            print("PASS")
        else:
            print(f"FAIL: expected total={expected_limit}, got {total_mem}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"MemGetInfo total mismatch: {result.stdout}"


def test_mem_get_info_free_decreases(cts):
    """After allocating memory, hipMemGetInfo should report decreased free memory."""
    result = cts.run_hip_test("""
        from hip_helper import HIPRuntime

        hip = HIPRuntime()
        alloc_size = 256 * 1024 * 1024  # 256 MiB

        free_before, total_before = hip.mem_get_info()
        ptr = hip.malloc(alloc_size)
        free_after, total_after = hip.mem_get_info()

        print(f"free_before={free_before}")
        print(f"total_before={total_before}")
        print(f"free_after={free_after}")
        print(f"total_after={total_after}")

        # Total should remain the same
        total_ok = (total_before == total_after)
        # Free should decrease by the allocation size
        free_delta = free_before - free_after
        free_ok = (free_delta == alloc_size)

        hip.free(ptr)

        # After free, free memory should return to original
        free_restored, _ = hip.mem_get_info()
        print(f"free_restored={free_restored}")
        restore_ok = (free_restored == free_before)

        if total_ok and free_ok and restore_ok:
            print("PASS")
        else:
            print(f"FAIL: total_ok={total_ok} free_ok={free_ok} (delta={free_delta}, expected={alloc_size}) restore_ok={restore_ok}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"MemGetInfo free decrease test failed: {result.stdout}"


def test_device_total_mem(cts):
    """hipDeviceTotalMem should return the SHM-configured mem_limit."""
    result = cts.run_hip_test("""
        from hip_helper import HIPRuntime

        hip = HIPRuntime()
        total = hip.device_total_mem(0)

        print(f"total={total}")

        expected_limit = 1024 * 1024 * 1024  # 1 GiB
        if total == expected_limit:
            print("PASS")
        else:
            print(f"FAIL: expected {expected_limit}, got {total}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"DeviceTotalMem mismatch: {result.stdout}"


def test_device_properties_total_mem(cts):
    """hipGetDeviceProperties should report totalGlobalMem == SHM-configured mem_limit.

    PyTorch's caching allocator reads device_prop.totalGlobalMem (not hipMemGetInfo)
    to compute the byte limit for set_per_process_memory_fraction(). If this field
    isn't spoofed, fraction-based memory limits compute against the real GPU memory
    (e.g. 256 GiB) instead of the pod limit, breaking OOM tests.
    """
    result = cts.run_hip_test("""
        from hip_helper import HIPRuntime

        hip = HIPRuntime()
        total = hip.get_device_properties_total_mem(0)

        print(f"total_global_mem={total}")

        expected_limit = 1024 * 1024 * 1024  # 1 GiB
        if total == expected_limit:
            print("PASS")
        else:
            print(f"FAIL: expected {expected_limit}, got {total}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"DeviceProperties totalGlobalMem mismatch: {result.stdout}"


def test_spoofed_values_consistent(cts):
    """hipMemGetInfo total, hipDeviceTotalMem, and hipGetDeviceProperties.totalGlobalMem
    should all return the same spoofed value."""
    result = cts.run_hip_test("""
        from hip_helper import HIPRuntime

        hip = HIPRuntime()

        _, mem_get_info_total = hip.mem_get_info()
        device_total = hip.device_total_mem(0)
        props_total = hip.get_device_properties_total_mem(0)

        print(f"mem_get_info_total={mem_get_info_total}")
        print(f"device_total={device_total}")
        print(f"props_total={props_total}")

        if mem_get_info_total == device_total == props_total:
            print("PASS")
        else:
            print(f"FAIL: memGetInfo={mem_get_info_total}, deviceTotalMem={device_total}, props={props_total}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Consistency test failed: {result.stdout}"


def test_spoofed_with_custom_limit(cts_factory):
    """Verify spoofing works with a non-default memory limit (512 MiB)."""
    custom_limit = 512 * MiB
    cts = cts_factory(
        devices=[
            DeviceSpec(
                uuid=DEFAULT_TEST_UUID,
                mem_limit=custom_limit,
                device_idx=0,
            )
        ]
    )

    result = cts.run_hip_test(f"""
        from hip_helper import HIPRuntime

        hip = HIPRuntime()
        expected = {custom_limit}

        _, total_from_info = hip.mem_get_info()
        total_from_dev = hip.device_total_mem(0)

        print(f"mem_get_info_total={{total_from_info}}")
        print(f"device_total={{total_from_dev}}")

        if total_from_info == expected and total_from_dev == expected:
            print("PASS")
        else:
            print(f"FAIL: expected {{expected}}, info={{total_from_info}}, dev={{total_from_dev}}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Custom limit spoofing failed: {result.stdout}"


class TestUnmappedDeviceFallthrough:
    """When hipDeviceTotalMem is called with a device not in the pod's SHM config,
    the limiter falls through to the native call, returning the real GPU total.

    This could leak real GPU memory size to a workload. The test verifies the
    current behavior (fallthrough) so any future change to deny instead would
    be caught.

    Requires a system with at least 2 AMD GPUs.
    """

    def test_device_total_mem_unmapped_returns_real(self, cts_factory):
        """Configure SHM with only device 0. hipDeviceTotalMem(1) should return
        the real GPU total (not the pod limit), because device 1 is not in SHM."""
        fixture = cts_factory(
            devices=[
                DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=512 * MiB, device_idx=0),
                # Device 1 intentionally NOT configured in SHM
            ],
        )

        configured_limit = 512 * MiB
        script = f"""\
from hip_helper import HIPRuntime
hip = HIPRuntime()

count = hip.get_device_count()
print(f"DEVICE_COUNT={{count}}")
if count < 2:
    print("SKIP")
    exit(0)

# Device 0 — configured, should return spoofed limit
total_0 = hip.device_total_mem(0)
print(f"TOTAL_0={{total_0}}")

# Device 1 — NOT configured in SHM, limiter should fall through to native
total_1 = hip.device_total_mem(1)
print(f"TOTAL_1={{total_1}}")

configured_limit = {configured_limit}
if total_0 == configured_limit:
    print("DEVICE_0_SPOOFED")
if total_1 != configured_limit:
    print("DEVICE_1_NOT_SPOOFED")
"""
        result = fixture.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"

        if "SKIP" in result.stdout:
            pytest.skip("System has fewer than 2 AMD GPUs")

        assert "DEVICE_0_SPOOFED" in result.stdout, (
            f"Device 0 should be spoofed to configured limit, got:\n{result.stdout}"
        )
        assert "DEVICE_1_NOT_SPOOFED" in result.stdout, (
            f"Device 1 (unmapped) should return real GPU total, not the pod limit. "
            f"Got:\n{result.stdout}"
        )

    def test_mem_get_info_unmapped_device(self, cts_factory):
        """Configure SHM with only device 0. After hipSetDevice(1), hipMemGetInfo
        should fall through to native since device 1 is not in the limiter's mapping."""
        fixture = cts_factory(
            devices=[
                DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=512 * MiB, device_idx=0),
            ],
        )

        configured_limit = 512 * MiB
        script = f"""\
from hip_helper import HIPRuntime
hip = HIPRuntime()

count = hip.get_device_count()
print(f"DEVICE_COUNT={{count}}")
if count < 2:
    print("SKIP")
    exit(0)

# Device 0 — configured
hip.set_device(0)
free_0, total_0 = hip.mem_get_info()
print(f"TOTAL_0={{total_0}}")

# Device 1 — not configured in SHM
hip.set_device(1)
free_1, total_1 = hip.mem_get_info()
print(f"TOTAL_1={{total_1}}")

configured_limit = {configured_limit}
if total_0 == configured_limit:
    print("DEVICE_0_SPOOFED")
if total_1 != configured_limit:
    print("DEVICE_1_NOT_SPOOFED")
"""
        result = fixture.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"

        if "SKIP" in result.stdout:
            pytest.skip("System has fewer than 2 AMD GPUs")

        assert "DEVICE_0_SPOOFED" in result.stdout, (
            f"Device 0 should be spoofed, got:\n{result.stdout}"
        )
        assert "DEVICE_1_NOT_SPOOFED" in result.stdout, (
            f"Device 1 (unmapped) hipMemGetInfo should return real total, "
            f"got:\n{result.stdout}"
        )

    def test_device_properties_unmapped_returns_real(self, cts_factory):
        """Configure SHM with only device 0. hipGetDeviceProperties(1).totalGlobalMem
        should fall through to native since device 1 is not in the limiter's mapping."""
        fixture = cts_factory(
            devices=[
                DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=512 * MiB, device_idx=0),
            ],
        )

        configured_limit = 512 * MiB
        script = f"""\
from hip_helper import HIPRuntime
hip = HIPRuntime()

count = hip.get_device_count()
print(f"DEVICE_COUNT={{count}}")
if count < 2:
    print("SKIP")
    exit(0)

# Device 0 — configured
props_0 = hip.get_device_properties_total_mem(0)
print(f"PROPS_0={{props_0}}")

# Device 1 — not configured in SHM
props_1 = hip.get_device_properties_total_mem(1)
print(f"PROPS_1={{props_1}}")

configured_limit = {configured_limit}
if props_0 == configured_limit:
    print("DEVICE_0_SPOOFED")
if props_1 != configured_limit:
    print("DEVICE_1_NOT_SPOOFED")
"""
        result = fixture.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"

        if "SKIP" in result.stdout:
            pytest.skip("System has fewer than 2 AMD GPUs")

        assert "DEVICE_0_SPOOFED" in result.stdout, (
            f"Device 0 properties should be spoofed, got:\n{result.stdout}"
        )
        assert "DEVICE_1_NOT_SPOOFED" in result.stdout, (
            f"Device 1 (unmapped) hipGetDeviceProperties should return real total, "
            f"got:\n{result.stdout}"
        )


class TestMultiDeviceSpoofing:
    """Verify hipDeviceTotalMem and hipMemGetInfo return per-device limits when
    multiple devices are configured with different memory limits.

    Requires a system with at least 2 AMD GPUs."""

    def test_device_total_mem_per_device(self, cts_factory):
        """hipDeviceTotalMem(0) and hipDeviceTotalMem(1) should return their
        respective SHM-configured mem_limits."""
        uuid_0 = _discover_gpu_pci_bus_id(0)
        uuid_1 = _discover_gpu_pci_bus_id(1)
        limit_0 = 512 * MiB
        limit_1 = 256 * MiB

        fixture = cts_factory(
            devices=[
                DeviceSpec(uuid=uuid_0, mem_limit=limit_0, device_idx=0),
                DeviceSpec(uuid=uuid_1, mem_limit=limit_1, device_idx=1),
            ],
        )

        script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS

hip = HIPRuntime()
count = hip.get_device_count()
print(f"DEVICE_COUNT={{count}}")

if count < 2:
    print("SKIP_NOT_ENOUGH_DEVICES")
    exit(0)

total_0 = hip.device_total_mem(0)
total_1 = hip.device_total_mem(1)
print(f"TOTAL_0={{total_0}}")
print(f"TOTAL_1={{total_1}}")
"""
        result = fixture.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"

        if "SKIP_NOT_ENOUGH_DEVICES" in result.stdout:
            pytest.skip("System has fewer than 2 AMD GPUs")

        values = parse_kv_output(result.stdout)
        assert values["TOTAL_0"] == limit_0, (
            f"Device 0 total ({values['TOTAL_0']}) != limit ({limit_0})"
        )
        assert values["TOTAL_1"] == limit_1, (
            f"Device 1 total ({values['TOTAL_1']}) != limit ({limit_1})"
        )

    def test_mem_get_info_per_device(self, cts_factory):
        """hipMemGetInfo should report the correct total for whichever device
        is currently set via hipSetDevice."""
        uuid_0 = _discover_gpu_pci_bus_id(0)
        uuid_1 = _discover_gpu_pci_bus_id(1)
        limit_0 = 512 * MiB
        limit_1 = 256 * MiB

        fixture = cts_factory(
            devices=[
                DeviceSpec(uuid=uuid_0, mem_limit=limit_0, device_idx=0),
                DeviceSpec(uuid=uuid_1, mem_limit=limit_1, device_idx=1),
            ],
        )

        script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS

hip = HIPRuntime()
count = hip.get_device_count()
print(f"DEVICE_COUNT={{count}}")

if count < 2:
    print("SKIP_NOT_ENOUGH_DEVICES")
    exit(0)

hip.set_device(0)
free_0, total_0 = hip.mem_get_info()
print(f"TOTAL_0={{total_0}}")
print(f"FREE_0={{free_0}}")

hip.set_device(1)
free_1, total_1 = hip.mem_get_info()
print(f"TOTAL_1={{total_1}}")
print(f"FREE_1={{free_1}}")
"""
        result = fixture.run_hip_test(script)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"

        if "SKIP_NOT_ENOUGH_DEVICES" in result.stdout:
            pytest.skip("System has fewer than 2 AMD GPUs")

        values = parse_kv_output(result.stdout)
        assert values["TOTAL_0"] == limit_0, (
            f"Device 0 memGetInfo total ({values['TOTAL_0']}) != limit ({limit_0})"
        )
        assert values["TOTAL_1"] == limit_1, (
            f"Device 1 memGetInfo total ({values['TOTAL_1']}) != limit ({limit_1})"
        )
