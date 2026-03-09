"""
CTS Conftest — Pytest fixtures for subprocess-based testing with LD_PRELOAD.

Each test spawns a fresh subprocess with the hip-limiter loaded via LD_PRELOAD,
ensuring a clean limiter instance per test. The fixture creates a temporary SHM
file, manages a background heartbeat updater, and provides helpers to run HIP
test scripts and inspect SHM accounting.

Key environment variables set for each subprocess:
  LD_PRELOAD        — path to libhip_limiter.so
  TF_SHM_FILE       — path to the test SHM file (triggers mock mode)
  TF_VISIBLE_DEVICES — comma-separated device UUIDs
  ENABLE_HIP_HOOKS  — "true"
  RUST_LOG          — "hip_limiter=debug"

Design decision — inline subprocess scripts:
  Test scripts are passed as inline strings rather than separate .py files.
  This is intentional: each subprocess gets a fresh LD_PRELOAD + limiter init,
  so the "test logic" is the script content itself. Inline strings keep the
  assertion context (what the script does) co-located with the pytest assertion
  (what we expect), making each test self-contained and readable top-to-bottom.
  Extracting scripts to files would scatter the test logic across two locations
  with no isolation benefit (the subprocess boundary already provides that).
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from shm_writer import (
    DeviceSpec,
    create_shm_file,
    read_device_count,
    read_device_uuid,
    read_heartbeat,
    read_mem_limit,
    read_pod_memory_used,
    update_heartbeat,
)

# ── Shared constants ──
KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB


def parse_kv_output(stdout: str) -> dict:
    """Parse KEY=VALUE lines from subprocess stdout into a dict with int values."""
    values = {}
    for line in stdout.strip().split("\n"):
        if "=" in line:
            key, val = line.split("=", 1)
            try:
                values[key.strip()] = int(val.strip())
            except ValueError:
                values[key.strip()] = val.strip()
    return values


def _has_amd_gpu() -> bool:
    """Check if an AMD GPU is accessible via HIP."""
    try:
        import ctypes
        hip = ctypes.CDLL("libamdhip64.so")
        count = ctypes.c_int(0)
        err = hip.hipGetDeviceCount(ctypes.byref(count))
        return err == 0 and count.value > 0
    except (OSError, Exception):
        return False


HAS_AMD_GPU = _has_amd_gpu()

requires_gpu = pytest.mark.skipif(not HAS_AMD_GPU, reason="No AMD GPU available")


# ── Configuration ──

# Default path to the hip-limiter shared library.
# Can be overridden via HIP_LIMITER_LIB env var.
DEFAULT_HIP_LIMITER_LIB = os.environ.get(
    "HIP_LIMITER_LIB",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "target",
        "release",
        "libhip_limiter.so",
    ),
)

# Default path to the HIP runtime library.
DEFAULT_HIP_LIB_PATH = os.environ.get(
    "HIP_LIB_PATH",
    os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib", "libamdhip64.so"),
)

def _discover_gpu_pci_bus_id(device_index: int = 0) -> str:
    """Discover the real PCI bus ID for a GPU device via HIP runtime."""
    import ctypes
    try:
        hip = ctypes.CDLL("libamdhip64.so")
        bus_id = ctypes.create_string_buffer(64)
        err = hip.hipDeviceGetPCIBusId(bus_id, 64, device_index)
        if err == 0 and bus_id.value:
            return bus_id.value.decode().strip()
    except (OSError, Exception):
        pass
    return "0000:03:00.0"  # fallback for environments without HIP


# Default test device UUID (auto-discovered from device 0's PCI bus ID)
DEFAULT_TEST_UUID = os.environ.get("CTS_TEST_UUID", _discover_gpu_pci_bus_id(0))

# Default memory limit for tests (1 GiB)
DEFAULT_MEM_LIMIT = int(os.environ.get("CTS_MEM_LIMIT", str(1 * 1024 * 1024 * 1024)))

# Heartbeat interval for background updater (seconds)
HEARTBEAT_INTERVAL = 0.5

# Subprocess timeout (seconds)
SUBPROCESS_TIMEOUT = int(os.environ.get("CTS_SUBPROCESS_TIMEOUT", "30"))


@dataclass
class SHMState:
    """Snapshot of SHM accounting state for verification."""

    device_count: int
    devices: Dict[int, dict] = field(default_factory=dict)
    heartbeat: int = 0


@dataclass
class SubprocessResult:
    """Result from running a test subprocess."""

    returncode: int
    stdout: str
    stderr: str
    output: str  # Combined stdout + stderr for convenience

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


class HeartbeatUpdater:
    """Background thread that periodically updates the SHM heartbeat.

    NOTE: The heartbeat is written via file I/O (open + seek + write) while the
    Rust limiter reads via mmap. This works on Linux because the unified page
    cache ensures coherence between file I/O and mmap on the same file. This
    assumption does not hold on all operating systems.
    """

    def __init__(self, shm_path: str, interval: float = HEARTBEAT_INTERVAL):
        self._shm_path = shm_path
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                update_heartbeat(self._shm_path)
            except (FileNotFoundError, OSError):
                pass  # SHM file may be cleaned up during teardown
            self._stop_event.wait(self._interval)


class CTSFixture:
    """Test fixture providing SHM setup, subprocess launching, and accounting verification.

    Usage in tests:
        def test_something(cts):
            result = cts.run_hip_test('''
                from hip_helper import HIPRuntime, HIP_SUCCESS
                hip = HIPRuntime()
                ptr = hip.malloc(1024)
                hip.free(ptr)
                print("PASS")
            ''')
            assert result.succeeded
            assert "PASS" in result.stdout
    """

    def __init__(
        self,
        devices: Optional[List[DeviceSpec]] = None,
        limiter_lib: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        enable_hooks: bool = True,
        heartbeat: bool = True,
    ):
        self._limiter_lib = limiter_lib or DEFAULT_HIP_LIMITER_LIB
        self._extra_env = extra_env or {}
        self._enable_hooks = enable_hooks
        self._heartbeat_enabled = heartbeat

        # Default: single device with 1 GiB limit
        if devices is None:
            devices = [
                DeviceSpec(
                    uuid=DEFAULT_TEST_UUID,
                    mem_limit=DEFAULT_MEM_LIMIT,
                    device_idx=0,
                )
            ]
        self._devices = devices

        self._tmpdir: Optional[str] = None
        self._shm_path: Optional[str] = None
        self._heartbeat_updater: Optional[HeartbeatUpdater] = None

    def setup(self):
        """Create temp directory, write SHM, start heartbeat."""
        self._tmpdir = tempfile.mkdtemp(prefix="cts_")
        self._shm_path = os.path.join(self._tmpdir, "shm")

        create_shm_file(self._shm_path, self._devices)

        if self._heartbeat_enabled:
            self._heartbeat_updater = HeartbeatUpdater(self._shm_path)
            self._heartbeat_updater.start()

    def teardown(self):
        """Stop heartbeat, clean up temp directory."""
        if self._heartbeat_updater is not None:
            self._heartbeat_updater.stop()
            self._heartbeat_updater = None

        if self._tmpdir is not None and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    @property
    def shm_path(self) -> str:
        """Path to the SHM file."""
        assert self._shm_path is not None, "Fixture not set up"
        return self._shm_path

    @property
    def tmpdir(self) -> str:
        """Path to the temporary directory."""
        assert self._tmpdir is not None, "Fixture not set up"
        return self._tmpdir

    def _build_env(self, extra_env: Optional[Dict[str, str]] = None) -> dict:
        """Build the subprocess environment dictionary."""
        env = os.environ.copy()

        # Core env vars for the limiter
        env["LD_PRELOAD"] = self._limiter_lib
        env["TF_SHM_FILE"] = self._shm_path
        env["ENABLE_HIP_HOOKS"] = "true" if self._enable_hooks else "false"
        env["RUST_LOG"] = env.get("RUST_LOG", "hip_limiter=debug")

        # Set TF_VISIBLE_DEVICES from device UUIDs
        uuids = [d.uuid for d in self._devices]
        env["TF_VISIBLE_DEVICES"] = ",".join(uuids)

        # Add PYTHONPATH so subprocess can import hip_helper and shm_writer
        cts_dir = os.path.dirname(os.path.abspath(__file__))
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{cts_dir}:{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = cts_dir

        # Apply fixture-level extra env
        env.update(self._extra_env)

        # Apply call-level extra env (a value of None removes the key)
        if extra_env:
            for key, value in extra_env.items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value

        return env

    def _run_subprocess(
        self,
        script: str,
        timeout: int = SUBPROCESS_TIMEOUT,
        use_preload: bool = True,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> SubprocessResult:
        """Run a Python script as a subprocess with the configured SHM env vars.

        Args:
            script: Python source code to execute.
            timeout: Maximum execution time in seconds.
            use_preload: If True, include LD_PRELOAD (limiter active).
                         If False, exclude LD_PRELOAD (direct HIP calls).
            extra_env: Additional environment variables for this specific run.

        Returns:
            SubprocessResult with returncode, stdout, stderr.
        """
        assert self._tmpdir is not None, "Fixture not set up — call setup() first"

        suffix = "test_script.py" if use_preload else "test_script_raw.py"
        script_path = os.path.join(self._tmpdir, suffix)
        with open(script_path, "w") as f:
            f.write(textwrap.dedent(script))

        env = self._build_env(extra_env)
        if not use_preload:
            env.pop("LD_PRELOAD", None)

        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            return SubprocessResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                output=proc.stdout + proc.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return SubprocessResult(
                returncode=-1,
                stdout=exc.stdout.decode() if exc.stdout else "",
                stderr=exc.stderr.decode() if exc.stderr else f"TIMEOUT after {timeout}s",
                output=f"TIMEOUT after {timeout}s",
            )

    def run_hip_test(
        self,
        script: str,
        timeout: int = SUBPROCESS_TIMEOUT,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> SubprocessResult:
        """Run a Python script as a subprocess with LD_PRELOAD and SHM env vars.

        The hip-limiter is loaded via LD_PRELOAD, intercepting HIP API calls
        made by the script (via hip_helper.py or directly).
        """
        return self._run_subprocess(script, timeout, use_preload=True, extra_env=extra_env)

    def run_hip_test_raw(
        self,
        script: str,
        timeout: int = SUBPROCESS_TIMEOUT,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> SubprocessResult:
        """Like run_hip_test but WITHOUT LD_PRELOAD (direct HIP calls, no interception)."""
        return self._run_subprocess(script, timeout, use_preload=False, extra_env=extra_env)

    def read_shm_accounting(self) -> SHMState:
        """Read current SHM state for verification after test execution.

        Returns an SHMState snapshot with device_count, per-device accounting,
        and heartbeat timestamp.
        """
        state = SHMState(
            device_count=read_device_count(self._shm_path),
            heartbeat=read_heartbeat(self._shm_path),
        )

        for i in range(state.device_count):
            try:
                state.devices[i] = {
                    "uuid": read_device_uuid(self._shm_path, i),
                    "mem_limit": read_mem_limit(self._shm_path, i),
                    "pod_memory_used": read_pod_memory_used(self._shm_path, i),
                }
            except (FileNotFoundError, struct.error) as exc:
                print(f"WARNING: read_shm_accounting: device {i}: {exc}")

        return state

    def read_pod_memory_used(self, device_idx: int = 0) -> int:
        """Convenience: read pod_memory_used for a single device."""
        return read_pod_memory_used(self._shm_path, device_idx)


# ── Pytest Fixtures ──


@pytest.fixture
def cts():
    """Default CTS fixture: single device, 1 GiB memory limit."""
    fixture = CTSFixture()
    fixture.setup()
    yield fixture
    fixture.teardown()


@pytest.fixture
def cts_factory():
    """Factory fixture for creating CTS instances with custom configuration.

    Usage:
        def test_custom(cts_factory):
            cts = cts_factory(devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=2*GiB)])
            result = cts.run_hip_test("...")
    """
    fixtures = []

    def _factory(**kwargs) -> CTSFixture:
        fixture = CTSFixture(**kwargs)
        fixture.setup()
        fixtures.append(fixture)
        return fixture

    yield _factory

    for f in fixtures:
        f.teardown()


@pytest.fixture
def cts_no_heartbeat():
    """CTS fixture with heartbeat updater disabled (for testing stale heartbeat scenarios)."""
    fixture = CTSFixture(heartbeat=False)
    fixture.setup()
    yield fixture
    fixture.teardown()


@pytest.fixture
def cts_no_hooks():
    """CTS fixture with HIP hooks disabled (passthrough mode)."""
    fixture = CTSFixture(enable_hooks=False)
    fixture.setup()
    yield fixture
    fixture.teardown()
