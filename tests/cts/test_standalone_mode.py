"""
Standalone Mode Tests — Validates TF_MEMORY_LIMIT mode where the limiter creates its own SHM
without a hypervisor or pre-created SHM file.

Key behaviors under test:
  - Limiter initializes from TF_MEMORY_LIMIT env var alone (no TF_SHM_FILE, no hypervisor)
  - Memory allocation enforcement works (alloc within limit succeeds, over limit fails)
  - Info spoofing reports the configured limit
  - Free tracking decrements accounting correctly
  - Multiple allocations are tracked across a process lifetime
"""

import os
import sys
import subprocess
import tempfile
import textwrap

import pytest

from conftest import DEFAULT_HIP_LIMITER_LIB, SUBPROCESS_TIMEOUT, requires_gpu

pytestmark = requires_gpu


def _standalone_env(shm_dir: str, mem_limit: str = "1G", extra_env: dict = None):
    """Build env dict for standalone mode."""
    env = os.environ.copy()
    env["LD_PRELOAD"] = DEFAULT_HIP_LIMITER_LIB
    env["TF_MEMORY_LIMIT"] = mem_limit
    env["SHM_PATH"] = shm_dir
    env["ENABLE_HIP_HOOKS"] = "true"
    env["RUST_LOG"] = env.get("RUST_LOG", "hip_limiter=debug")

    # Ensure mock mode env vars are NOT set
    env.pop("TF_SHM_FILE", None)
    env.pop("TF_VISIBLE_DEVICES", None)
    env.pop("HYPERVISOR_IP", None)
    env.pop("HYPERVISOR_PORT", None)

    cts_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = cts_dir

    if extra_env:
        for key, value in extra_env.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value

    return env


def _run_standalone(script: str, mem_limit: str = "1G",
                    extra_env: dict = None, timeout: int = SUBPROCESS_TIMEOUT):
    """Run a script with TF_MEMORY_LIMIT standalone mode (no SHM file, no hypervisor)."""
    with tempfile.TemporaryDirectory(prefix="cts_standalone_") as tmpdir:
        shm_dir = os.path.join(tmpdir, "shm")
        os.makedirs(shm_dir, exist_ok=True)
        env = _standalone_env(shm_dir, mem_limit, extra_env)

        cts_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(tmpdir, "test_script.py")
        with open(script_path, "w") as f:
            f.write(textwrap.dedent(script))

        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=cts_dir,
            )
            return proc
        except subprocess.TimeoutExpired as e:
            pytest.fail(
                f"Standalone test timed out after {timeout}s\n"
                f"stdout: {e.stdout}\nstderr: {e.stderr}"
            )


class TestStandaloneAllocation:
    """Verify allocation enforcement works in standalone mode."""

    def test_alloc_within_limit(self):
        """A small allocation should succeed when under the limit."""
        proc = _run_standalone("""
            from hip_helper import HIPRuntime

            hip = HIPRuntime()
            ptr = hip.malloc(1024 * 1024)  # 1 MiB
            assert ptr != 0, "malloc returned null"
            hip.free(ptr)
            print("PASS")
        """, mem_limit="1G")
        assert proc.returncode == 0, f"Failed: {proc.stderr}"
        assert "PASS" in proc.stdout, f"Test failed: {proc.stdout}"

    def test_alloc_over_limit_denied(self):
        """An allocation exceeding the limit should be denied with OOM."""
        proc = _run_standalone("""
            from hip_helper import HIPRuntime, HIP_ERROR_OUT_OF_MEMORY

            hip = HIPRuntime()
            # Try to allocate 2 GiB with a 1 GiB limit
            err, ptr = hip.malloc_raw(2 * 1024 * 1024 * 1024)
            if err == HIP_ERROR_OUT_OF_MEMORY:
                print("PASS")
            else:
                print(f"FAIL: expected OOM error ({HIP_ERROR_OUT_OF_MEMORY}), got err={err}")
        """, mem_limit="1G")
        assert proc.returncode == 0, f"Failed: {proc.stderr}"
        assert "PASS" in proc.stdout, f"Test failed: {proc.stdout}"

    def test_alloc_free_alloc_cycle(self):
        """Allocate, free, allocate again — accounting should allow reuse."""
        proc = _run_standalone("""
            from hip_helper import HIPRuntime

            hip = HIPRuntime()
            # Allocate 512 MiB
            ptr = hip.malloc(512 * 1024 * 1024)
            assert ptr != 0, "first malloc returned null"

            # Free it
            hip.free(ptr)

            # Allocate 512 MiB again — should succeed since we freed
            ptr2 = hip.malloc(512 * 1024 * 1024)
            assert ptr2 != 0, "second malloc returned null"
            hip.free(ptr2)
            print("PASS")
        """, mem_limit="1G")
        assert proc.returncode == 0, f"Failed: {proc.stderr}"
        assert "PASS" in proc.stdout, f"Test failed: {proc.stdout}"


class TestStandaloneSpoofing:
    """Verify info spoofing reports the standalone limit."""

    def test_mem_get_info_reports_limit(self):
        """hipMemGetInfo should report total == TF_MEMORY_LIMIT."""
        proc = _run_standalone("""
            from hip_helper import HIPRuntime

            hip = HIPRuntime()
            free_mem, total_mem = hip.mem_get_info()

            expected = 1000000000  # 1G = 1,000,000,000 bytes (SI)
            print(f"total={total_mem}")
            print(f"expected={expected}")
            if total_mem == expected:
                print("PASS")
            else:
                print(f"FAIL: expected total={expected}, got {total_mem}")
        """, mem_limit="1G")
        assert proc.returncode == 0, f"Failed: {proc.stderr}"
        assert "PASS" in proc.stdout, f"Spoofing mismatch: {proc.stdout}"

    def test_device_total_mem_reports_limit(self):
        """hipDeviceTotalMem should report TF_MEMORY_LIMIT."""
        proc = _run_standalone("""
            from hip_helper import HIPRuntime

            hip = HIPRuntime()
            total = hip.device_total_mem(0)

            expected = 1000000000  # 1G
            print(f"total={total}")
            if total == expected:
                print("PASS")
            else:
                print(f"FAIL: expected {expected}, got {total}")
        """, mem_limit="1G")
        assert proc.returncode == 0, f"Failed: {proc.stderr}"
        assert "PASS" in proc.stdout, f"Spoofing mismatch: {proc.stdout}"

    def test_free_decreases_after_alloc(self):
        """hipMemGetInfo free should decrease after allocation."""
        proc = _run_standalone("""
            from hip_helper import HIPRuntime

            hip = HIPRuntime()
            free_before, total = hip.mem_get_info()

            ptr = hip.malloc(100 * 1024 * 1024)  # 100 MB
            assert ptr != 0

            free_after, _ = hip.mem_get_info()
            hip.free(ptr)

            decrease = free_before - free_after
            print(f"free_before={free_before}")
            print(f"free_after={free_after}")
            print(f"decrease={decrease}")
            # Decrease should be at least 100 MB
            if decrease >= 100 * 1024 * 1024:
                print("PASS")
            else:
                print(f"FAIL: expected decrease >= 100MB, got {decrease}")
        """, mem_limit="1G")
        assert proc.returncode == 0, f"Failed: {proc.stderr}"
        assert "PASS" in proc.stdout, f"Free tracking failed: {proc.stdout}"


class TestStandaloneSizeFormats:
    """Verify different TF_MEMORY_LIMIT format strings work."""

    def test_gib_format(self):
        """126GiB should be parsed correctly."""
        proc = _run_standalone("""
            from hip_helper import HIPRuntime

            hip = HIPRuntime()
            _, total = hip.mem_get_info()

            expected = 126 * 1024 * 1024 * 1024  # 126 GiB
            print(f"total={total}")
            if total == expected:
                print("PASS")
            else:
                print(f"FAIL: expected {expected}, got {total}")
        """, mem_limit="126GiB")
        assert proc.returncode == 0, f"Failed: {proc.stderr}"
        assert "PASS" in proc.stdout, f"Size parsing failed: {proc.stdout}"

    def test_plain_bytes_format(self):
        """Plain byte count should work."""
        proc = _run_standalone("""
            from hip_helper import HIPRuntime

            hip = HIPRuntime()
            _, total = hip.mem_get_info()

            expected = 1073741824  # exactly 1 GiB in bytes
            print(f"total={total}")
            if total == expected:
                print("PASS")
            else:
                print(f"FAIL: expected {expected}, got {total}")
        """, mem_limit="1073741824")
        assert proc.returncode == 0, f"Failed: {proc.stderr}"
        assert "PASS" in proc.stdout, f"Byte parsing failed: {proc.stdout}"


class TestStandaloneEdgeCases:
    """Edge cases for standalone mode."""

    def test_invalid_limit_passthrough(self):
        """Invalid TF_MEMORY_LIMIT should cause limiter to not init (passthrough)."""
        proc = _run_standalone("""
            from hip_helper import HIPRuntime

            hip = HIPRuntime()
            # With no limiter active, a normal allocation should succeed
            ptr = hip.malloc(1024)
            assert ptr != 0, "malloc failed"
            hip.free(ptr)
            print("PASS")
        """, mem_limit="not_a_number", extra_env={"TF_LOG_PATH": "stderr"})
        assert proc.returncode == 0, f"Failed: {proc.stderr}"
        assert "PASS" in proc.stdout, f"Passthrough failed: {proc.stdout}"
        # Verify the limiter logged the parse error (not silently ignored)
        output = proc.stdout + proc.stderr
        assert "invalid TF_MEMORY_LIMIT" in output, \
            f"Expected parse error in logs:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"

    def test_zero_limit_passthrough(self):
        """TF_MEMORY_LIMIT=0G is unparseable (zero value), so limiter does not init."""
        proc = _run_standalone("""
            from hip_helper import HIPRuntime

            hip = HIPRuntime()
            ptr = hip.malloc(1024)
            assert ptr != 0, "malloc failed"
            hip.free(ptr)
            print("PASS")
        """, mem_limit="0G", extra_env={"TF_LOG_PATH": "stderr"})
        assert proc.returncode == 0, f"Failed: {proc.stderr}"
        assert "PASS" in proc.stdout, f"Passthrough failed: {proc.stdout}"
        # Zero is treated as invalid by the parser
        output = proc.stdout + proc.stderr
        assert "invalid TF_MEMORY_LIMIT" in output, \
            f"Expected parse error in logs:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
