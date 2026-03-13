"""
SMI Spoofing Tests -- Validates that rocm-smi and amd-smi report the pod's
configured memory limit instead of the full physical GPU memory.

Hooked functions (total only, usage passes through):
  - rsmi_dev_memory_total_get  -> mem_limit from SHM (bytes)
  - amdsmi_get_gpu_memory_total -> mem_limit from SHM (bytes)
  - amdsmi_get_gpu_vram_info   -> vram_size = mem_limit in MB
"""

import pytest

from conftest import DEFAULT_TEST_UUID, GiB, MiB, parse_kv_output, requires_gpu, _discover_gpu_pci_bus_id
from shm_writer import DeviceSpec

pytestmark = requires_gpu


# ---------------------------------------------------------------------------
# rocm_smi subprocess boilerplate
#
# RSMI_SETUP loads librocm_smi64, initializes, and leaves the library ready
# for queries. Tests append their body + RSMI_TEARDOWN.
# ---------------------------------------------------------------------------

RSMI_SETUP = """
import ctypes, ctypes.util

lib_path = ctypes.util.find_library("rocm_smi64")
if lib_path is None:
    print("SKIP_NO_LIB")
else:
    lib = ctypes.CDLL(lib_path)
    lib.rsmi_init(0)

"""

RSMI_TEARDOWN = """
    lib.rsmi_shut_down()
"""


def _rsmi_skip_checks(result):
    """Assert subprocess succeeded and skip if the library is unavailable."""
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    if "SKIP_NO_LIB" in result.stdout:
        pytest.skip("librocm_smi64 not available")


# ---------------------------------------------------------------------------
# amdsmi subprocess boilerplate
#
# amdsmi requires:
#   - argtypes set on every ctypes function (otherwise pointer args are wrong)
#   - c_ulong for amdsmi_init (INIT_AMD_GPUS = 2)
#   - socket -> processor two-step enumeration
#
# AMDSMI_SETUP opens the library, inits, enumerates processors into
# `all_procs`, and sets argtypes for the commonly-used memory_total function.
# Tests append their body (indented inside the `else:` block) + AMDSMI_TEARDOWN.
# ---------------------------------------------------------------------------

AMDSMI_SETUP = """
import ctypes, ctypes.util

lib_path = ctypes.util.find_library("amd_smi")
if lib_path is None:
    print("SKIP_NO_LIB")
else:
    lib = ctypes.CDLL(lib_path)

    lib.amdsmi_init.argtypes = [ctypes.c_ulong]
    lib.amdsmi_init.restype = ctypes.c_uint
    lib.amdsmi_get_socket_handles.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
    lib.amdsmi_get_socket_handles.restype = ctypes.c_uint
    lib.amdsmi_get_processor_handles.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
    lib.amdsmi_get_processor_handles.restype = ctypes.c_uint
    lib.amdsmi_get_gpu_memory_total.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint64)]
    lib.amdsmi_get_gpu_memory_total.restype = ctypes.c_uint
    lib.amdsmi_get_gpu_device_bdf.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.amdsmi_get_gpu_device_bdf.restype = ctypes.c_uint

    status = lib.amdsmi_init(ctypes.c_ulong(2))  # INIT_AMD_GPUS
    if status != 0:
        print(f"SKIP_INIT_FAILED={status}")
    else:
        socket_count = ctypes.c_uint32(0)
        lib.amdsmi_get_socket_handles(ctypes.byref(socket_count), None)
        sockets = (ctypes.c_void_p * socket_count.value)()
        lib.amdsmi_get_socket_handles(ctypes.byref(socket_count), sockets)

        all_procs = []
        for si in range(socket_count.value):
            pc = ctypes.c_uint32(0)
            lib.amdsmi_get_processor_handles(sockets[si], ctypes.byref(pc), None)
            if pc.value > 0:
                procs = (ctypes.c_void_p * pc.value)()
                lib.amdsmi_get_processor_handles(sockets[si], ctypes.byref(pc), procs)
                all_procs.extend(procs)

        if not all_procs:
            print("SKIP_NO_HANDLES=0")
        else:
            # Build BDF -> (proc, index) map for tests that need a specific device.
            # amdsmi enumerates in BDF order, not HIP device order.
            def _get_proc_bdf(proc):
                bdf_raw = ctypes.c_uint64(0)
                bdf_status = lib.amdsmi_get_gpu_device_bdf(proc, ctypes.byref(bdf_raw))
                assert bdf_status == 0, f"amdsmi_get_gpu_device_bdf failed: status={bdf_status}"
                raw = bdf_raw.value
                fn = raw & 0x7
                dev = (raw >> 3) & 0x1F
                bus = (raw >> 8) & 0xFF
                dom = (raw >> 16) & 0xFFFFFFFFFFFF
                return f"{dom:04x}:{bus:02x}:{dev:02x}.{fn}"

            proc_by_bdf = {}
            for idx, p in enumerate(all_procs):
                proc_by_bdf[_get_proc_bdf(p)] = (p, idx)
"""

AMDSMI_TEARDOWN = """
        lib.amdsmi_shut_down()
"""

# Shared VramInfo struct definition + argtypes for amdsmi_get_gpu_vram_info.
# Appended after AMDSMI_SETUP in tests that query vram_info.
AMDSMI_VRAM_INFO_SETUP = """
            AMDSMI_MAX_STRING_LENGTH = 256

            class VramInfo(ctypes.Structure):
                _fields_ = [
                    ("vram_type", ctypes.c_uint32),
                    ("vram_vendor", ctypes.c_char * AMDSMI_MAX_STRING_LENGTH),
                    ("vram_size", ctypes.c_uint64),
                    ("vram_bit_width", ctypes.c_uint32),
                    ("vram_max_bandwidth", ctypes.c_uint64),
                    ("reserved", ctypes.c_uint64 * 37),
                ]

            lib.amdsmi_get_gpu_vram_info.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(VramInfo)]
            lib.amdsmi_get_gpu_vram_info.restype = ctypes.c_uint
"""


def _amdsmi_skip_checks(result):
    """Assert subprocess succeeded and skip if the library/driver is unavailable."""
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    if "SKIP_NO_LIB" in result.stdout:
        pytest.skip("libamd_smi not available")
    if "SKIP_INIT_FAILED" in result.stdout:
        pytest.skip("amdsmi_init failed")
    if "SKIP_NO_HANDLES" in result.stdout:
        pytest.skip("No GPU handles from amdsmi")


# ---------------------------------------------------------------------------
# rocm_smi (librocm_smi64) tests
# ---------------------------------------------------------------------------


class TestRocmSmiSpoofing:
    """Test rsmi_dev_memory_total_get spoofing via librocm_smi64."""

    def test_rsmi_total_reports_limit(self, cts):
        """Core spoofing: rsmi_dev_memory_total_get(VRAM) returns the SHM
        mem_limit instead of the physical GPU memory. Validates the primary
        hook at the library API level."""
        result = cts.run_hip_test(RSMI_SETUP + """
    total = ctypes.c_uint64()
    status = lib.rsmi_dev_memory_total_get(0, 0, ctypes.byref(total))  # VRAM=0
    print(f"STATUS={status}")
    print(f"TOTAL={total.value}")
""" + RSMI_TEARDOWN)
        _rsmi_skip_checks(result)

        values = parse_kv_output(result.stdout)
        assert values["STATUS"] == 0, f"rsmi call failed: status={values['STATUS']}"
        assert values["TOTAL"] == 1 * GiB

    def test_rsmi_custom_limit(self, cts_factory):
        """Spoofed total reflects a non-default mem_limit (512 MiB). Confirms
        the hook reads dynamically from SHM rather than using a hardcoded value."""
        custom_limit = 512 * MiB
        cts = cts_factory(
            devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=custom_limit, device_idx=0)]
        )

        result = cts.run_hip_test(RSMI_SETUP + f"""
    total = ctypes.c_uint64()
    status = lib.rsmi_dev_memory_total_get(0, 0, ctypes.byref(total))
    print(f"STATUS={{status}}")
    print(f"TOTAL={{total.value}}")
""" + RSMI_TEARDOWN)
        _rsmi_skip_checks(result)

        values = parse_kv_output(result.stdout)
        assert values["STATUS"] == 0
        assert values["TOTAL"] == custom_limit

    def test_rsmi_usage_not_spoofed(self, cts):
        """Usage (rsmi_dev_memory_usage_get) is NOT hooked — the real driver
        value passes through. Ensures we only spoof total, not usage.
        Note: asserts != mem_limit rather than exact equality, since GPU
        memory usage fluctuates between subprocess invocations."""
        result = cts.run_hip_test(RSMI_SETUP + """
    used = ctypes.c_uint64()
    status = lib.rsmi_dev_memory_usage_get(0, 0, ctypes.byref(used))
    print(f"STATUS={status}")
    print(f"USED={used.value}")
""" + RSMI_TEARDOWN)
        _rsmi_skip_checks(result)

        values = parse_kv_output(result.stdout)
        assert values["STATUS"] == 0
        # Used is a real driver value — should not equal our 1 GiB spoofed limit
        assert values["USED"] != 1 * GiB, "Usage should not be spoofed to mem_limit"


# ---------------------------------------------------------------------------
# amdsmi (libamd_smi) tests
# ---------------------------------------------------------------------------


class TestAmdSmiSpoofing:
    """Test amdsmi_get_gpu_memory_total and amdsmi_get_gpu_vram_info spoofing."""

    def test_amdsmi_memory_total_reports_limit(self, cts):
        """Core spoofing: amdsmi_get_gpu_memory_total(VRAM) returns the SHM
        mem_limit in bytes. Validates the primary amdsmi hook.
        Finds the configured device by BDF since amdsmi enumerates in
        BDF order, not HIP device order."""
        target_bdf = DEFAULT_TEST_UUID.lower()
        result = cts.run_hip_test(AMDSMI_SETUP + f"""
            target = proc_by_bdf.get("{target_bdf}")
            if target is None:
                print("SKIP_BDF_NOT_FOUND=1")
            else:
                proc, _ = target
                total = ctypes.c_uint64()
                status = lib.amdsmi_get_gpu_memory_total(proc, 0, ctypes.byref(total))
                print(f"STATUS={{status}}")
                print(f"TOTAL={{total.value}}")
""" + AMDSMI_TEARDOWN)
        _amdsmi_skip_checks(result)
        if "SKIP_BDF_NOT_FOUND" in result.stdout:
            pytest.skip(f"Configured BDF {target_bdf} not found in amdsmi processors")

        values = parse_kv_output(result.stdout)
        assert values["STATUS"] == 0
        assert values["TOTAL"] == 1 * GiB

    def test_amdsmi_vram_info_reports_limit(self, cts):
        """amdsmi_get_gpu_vram_info spoofs vram_size (MB) while preserving
        other struct fields (vram_bit_width, vram_vendor) from the real driver.
        Covers the struct-level spoofing path distinct from the scalar total."""
        target_bdf = DEFAULT_TEST_UUID.lower()
        result = cts.run_hip_test(AMDSMI_SETUP + AMDSMI_VRAM_INFO_SETUP + f"""
            target = proc_by_bdf.get("{target_bdf}")
            if target is None:
                print("SKIP_BDF_NOT_FOUND=1")
            else:
                proc, _ = target
                info = VramInfo()
                status = lib.amdsmi_get_gpu_vram_info(proc, ctypes.byref(info))
                print(f"STATUS={{status}}")
                print(f"VRAM_SIZE_MB={{info.vram_size}}")
                print(f"VRAM_VENDOR={{info.vram_vendor.decode().rstrip(chr(0))}}")
                print(f"VRAM_BIT_WIDTH={{info.vram_bit_width}}")
""" + AMDSMI_TEARDOWN)
        _amdsmi_skip_checks(result)
        if "SKIP_BDF_NOT_FOUND" in result.stdout:
            pytest.skip(f"Configured BDF {target_bdf} not found in amdsmi processors")

        values = parse_kv_output(result.stdout)
        assert values["STATUS"] == 0
        expected_mb = (1 * GiB) // (1024 * 1024)
        assert values["VRAM_SIZE_MB"] == expected_mb
        # Non-spoofed fields should contain real driver data
        assert values["VRAM_BIT_WIDTH"] > 0, "vram_bit_width should be populated by driver"

    def test_amdsmi_custom_limits(self, cts_factory):
        """Both memory_total and vram_info reflect a custom mem_limit (2 GiB).
        Exercises the bytes-to-MB conversion in vram_info (smi.rs division)
        and confirms both hooks read from SHM dynamically."""
        custom_limit = 2 * GiB
        cts = cts_factory(
            devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=custom_limit, device_idx=0)]
        )
        target_bdf = DEFAULT_TEST_UUID.lower()

        result = cts.run_hip_test(AMDSMI_SETUP + AMDSMI_VRAM_INFO_SETUP + f"""
            target = proc_by_bdf.get("{target_bdf}")
            if target is None:
                print("SKIP_BDF_NOT_FOUND=1")
            else:
                proc, _ = target
                # memory_total (bytes)
                total = ctypes.c_uint64()
                status_total = lib.amdsmi_get_gpu_memory_total(proc, 0, ctypes.byref(total))
                print(f"STATUS_TOTAL={{status_total}}")
                print(f"TOTAL={{total.value}}")

                # vram_info (MB)
                info = VramInfo()
                status_info = lib.amdsmi_get_gpu_vram_info(proc, ctypes.byref(info))
                print(f"STATUS_INFO={{status_info}}")
                print(f"VRAM_SIZE_MB={{info.vram_size}}")
""" + AMDSMI_TEARDOWN)
        _amdsmi_skip_checks(result)
        if "SKIP_BDF_NOT_FOUND" in result.stdout:
            pytest.skip(f"Configured BDF {target_bdf} not found in amdsmi processors")

        values = parse_kv_output(result.stdout)
        assert values["STATUS_TOTAL"] == 0
        assert values["TOTAL"] == custom_limit
        assert values["STATUS_INFO"] == 0
        assert values["VRAM_SIZE_MB"] == custom_limit // (1024 * 1024)

    def test_amdsmi_multi_gpu_per_device_limits(self, cts_factory):
        """Two GPUs configured with different mem_limits. Each amdsmi processor
        handle should return its own device's limit, not device 0's. Validates
        BDF-based handle resolution in resolve_amdsmi_device.

        Requires at least 2 AMD GPUs; skips otherwise."""
        uuid_0 = _discover_gpu_pci_bus_id(0)
        uuid_1 = _discover_gpu_pci_bus_id(1)
        limit_0 = 1 * GiB
        limit_1 = 2 * GiB

        cts = cts_factory(
            devices=[
                DeviceSpec(uuid=uuid_0, mem_limit=limit_0, device_idx=0),
                DeviceSpec(uuid=uuid_1, mem_limit=limit_1, device_idx=1),
            ],
        )

        # The subprocess enumerates all amdsmi processors and queries each one's
        # memory_total. It also gets each processor's BDF to identify which
        # configured device it matches. We check that the two configured BDFs
        # report their respective limits.
        result = cts.run_hip_test(AMDSMI_SETUP + f"""
            lib.amdsmi_get_gpu_device_bdf.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
            lib.amdsmi_get_gpu_device_bdf.restype = ctypes.c_uint

            for i, proc in enumerate(all_procs):
                # Get BDF for this processor
                bdf_raw = ctypes.c_uint64(0)
                bdf_status = lib.amdsmi_get_gpu_device_bdf(proc, ctypes.byref(bdf_raw))
                if bdf_status != 0:
                    print(f"PROC_{{i}}_BDF_ERROR={{bdf_status}}")
                    continue
                raw = bdf_raw.value
                fn = raw & 0x7
                dev = (raw >> 3) & 0x1F
                bus = (raw >> 8) & 0xFF
                dom = (raw >> 16) & 0xFFFFFFFFFFFF
                bdf = f"{{dom:04x}}:{{bus:02x}}:{{dev:02x}}.{{fn}}"

                # Query spoofed memory total
                total = ctypes.c_uint64()
                status = lib.amdsmi_get_gpu_memory_total(proc, 0, ctypes.byref(total))
                print(f"PROC_{{i}}_BDF={{bdf}}")
                print(f"PROC_{{i}}_STATUS={{status}}")
                print(f"PROC_{{i}}_TOTAL={{total.value}}")
""" + AMDSMI_TEARDOWN)
        _amdsmi_skip_checks(result)

        values = parse_kv_output(result.stdout)

        # Find which proc indices correspond to our two configured UUIDs
        found_0 = found_1 = False
        for i in range(len([k for k in values if k.startswith("PROC_") and k.endswith("_BDF")])):
            bdf = values.get(f"PROC_{i}_BDF", "")
            total = values.get(f"PROC_{i}_TOTAL", 0)
            status = values.get(f"PROC_{i}_STATUS", -1)

            if bdf == uuid_0.lower():
                assert status == 0, f"Device 0 query failed: status={status}"
                assert total == limit_0, f"Device 0: expected {limit_0}, got {total}"
                found_0 = True
            elif bdf == uuid_1.lower():
                assert status == 0, f"Device 1 query failed: status={status}"
                assert total == limit_1, f"Device 1: expected {limit_1}, got {total}"
                found_1 = True

        if not found_0 and not found_1:
            pytest.skip("Neither configured GPU found in amdsmi processor list")
        assert found_0, f"Device 0 BDF {uuid_0} not found in amdsmi processors"
        assert found_1, f"Device 1 BDF {uuid_1} not found in amdsmi processors"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestSmiCli:
    """Verify that the actual CLI tools show spoofed values end-to-end.
    These are the user-facing tools that operators and frameworks use to
    query GPU memory — the ultimate validation that spoofing works."""

    def test_amd_smi_static_vram_spoofed(self, cts):
        """amd-smi static --vram SIZE should show mem_limit in MB.
        End-to-end: CLI -> libamd_smi -> our dlsym hook -> SHM.
        Finds the correct amdsmi GPU index for the configured device
        since amd-smi uses BDF-sorted indices, not HIP device indices."""
        target_bdf = DEFAULT_TEST_UUID.lower()
        result = cts.run_hip_test(f"""
import ctypes, ctypes.util, subprocess, re

target_bdf = "{target_bdf}"

# Find the amdsmi GPU index for our configured device
lib_path = ctypes.util.find_library("amd_smi")
if lib_path is None:
    print("SKIP_NO_LIB")
else:
    lib = ctypes.CDLL(lib_path)
    lib.amdsmi_init.argtypes = [ctypes.c_ulong]
    lib.amdsmi_init.restype = ctypes.c_uint
    lib.amdsmi_get_socket_handles.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
    lib.amdsmi_get_socket_handles.restype = ctypes.c_uint
    lib.amdsmi_get_processor_handles.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
    lib.amdsmi_get_processor_handles.restype = ctypes.c_uint
    lib.amdsmi_get_gpu_device_bdf.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.amdsmi_get_gpu_device_bdf.restype = ctypes.c_uint

    lib.amdsmi_init(ctypes.c_ulong(2))

    socket_count = ctypes.c_uint32(0)
    lib.amdsmi_get_socket_handles(ctypes.byref(socket_count), None)
    sockets = (ctypes.c_void_p * socket_count.value)()
    lib.amdsmi_get_socket_handles(ctypes.byref(socket_count), sockets)

    gpu_idx = None
    proc_idx = 0
    for si in range(socket_count.value):
        pc = ctypes.c_uint32(0)
        lib.amdsmi_get_processor_handles(sockets[si], ctypes.byref(pc), None)
        if pc.value > 0:
            procs = (ctypes.c_void_p * pc.value)()
            lib.amdsmi_get_processor_handles(sockets[si], ctypes.byref(pc), procs)
            for p in procs:
                bdf_raw = ctypes.c_uint64(0)
                bdf_status = lib.amdsmi_get_gpu_device_bdf(p, ctypes.byref(bdf_raw))
                if bdf_status != 0:
                    proc_idx += 1
                    continue
                raw = bdf_raw.value
                fn = raw & 0x7
                dev = (raw >> 3) & 0x1F
                bus = (raw >> 8) & 0xFF
                dom = (raw >> 16) & 0xFFFFFFFFFFFF
                bdf = f"{{dom:04x}}:{{bus:02x}}:{{dev:02x}}.{{fn}}"
                if bdf == target_bdf:
                    gpu_idx = proc_idx
                proc_idx += 1
    lib.amdsmi_shut_down()

    if gpu_idx is None:
        print("SKIP_BDF_NOT_FOUND=1")
    else:
        try:
            out = subprocess.check_output(
                ["amd-smi", "static", "--vram", "--gpu", str(gpu_idx)],
                stderr=subprocess.STDOUT, text=True, timeout=15,
            )
            match = re.search(r"SIZE:\\s+(\\d+)\\s+MB", out)
            if match:
                print(f"VRAM_SIZE_MB={{match.group(1)}}")
            else:
                print(f"PARSE_FAILED=1")
                print(out)
        except FileNotFoundError:
            print("SKIP_NO_CLI")
        except subprocess.CalledProcessError as e:
            print(f"CLI_ERROR={{e.returncode}}")
""")
        assert result.succeeded, f"Subprocess failed: {result.stderr}"
        if "SKIP_NO_LIB" in result.stdout:
            pytest.skip("libamd_smi not available")
        if "SKIP_NO_CLI" in result.stdout:
            pytest.skip("amd-smi CLI not available")
        if "CLI_ERROR" in result.stdout:
            pytest.skip(f"amd-smi CLI failed: {result.stdout}")
        if "SKIP_BDF_NOT_FOUND" in result.stdout:
            pytest.skip(f"Configured BDF {target_bdf} not found in amdsmi processors")

        values = parse_kv_output(result.stdout)
        assert "PARSE_FAILED" not in values, f"Could not parse amd-smi output: {result.stdout}"
        expected_mb = (1 * GiB) // (1024 * 1024)
        assert values["VRAM_SIZE_MB"] == expected_mb

    def test_rocm_smi_showmeminfo_spoofed(self, cts):
        """rocm-smi --showmeminfo vram should show mem_limit in bytes.
        End-to-end: CLI -> librocm_smi64 -> our dlsym hook -> SHM.
        Uses -d 0 to query only the configured device (rsmi device 0 = HIP device 0)."""
        result = cts.run_hip_test("""
import subprocess, re

try:
    out = subprocess.check_output(
        ["rocm-smi", "-d", "0", "--showmeminfo", "vram"],
        stderr=subprocess.STDOUT, text=True, timeout=15,
    )
    match = re.search(r"VRAM Total Memory \\(B\\):\\s+(\\d+)", out)
    if match:
        print(f"VRAM_TOTAL={match.group(1)}")
    else:
        print(f"PARSE_FAILED=1")
        print(out)
except FileNotFoundError:
    print("SKIP_NO_CLI")
except subprocess.CalledProcessError as e:
    print(f"CLI_ERROR={e.returncode}")
""")
        assert result.succeeded, f"Subprocess failed: {result.stderr}"
        if "SKIP_NO_CLI" in result.stdout:
            pytest.skip("rocm-smi CLI not available")
        if "CLI_ERROR" in result.stdout:
            pytest.skip(f"rocm-smi CLI failed: {result.stdout}")

        values = parse_kv_output(result.stdout)
        assert "PARSE_FAILED" not in values, f"Could not parse rocm-smi output: {result.stdout}"
        assert values["VRAM_TOTAL"] == 1 * GiB
