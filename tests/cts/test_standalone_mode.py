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


class TestStandaloneForkSafety:
    """Fork safety: the limiter must not initialize the HIP runtime during standalone init."""

    def test_fork_safety_child_hip_malloc(self):
        """Verify that fork() after limiter init doesn't break HIP in the child.

        This is the core fork-safety property: the limiter must NOT call
        hipGetDeviceCount/hipDeviceGetPCIBusId during init (standalone mode),
        so the child process can use HIP without inheriting stale GPU state.
        """
        script = """\
            import os
            import sys
            from hip_helper import HIPRuntime

            # Construct HIPRuntime in parent — this triggers limiter init via LD_PRELOAD
            hip = HIPRuntime()

            # Fork — child must be able to use HIP
            pid = os.fork()
            if pid == 0:
                # Child process: create a fresh HIPRuntime and use GPU
                try:
                    child_hip = HIPRuntime()
                    count = child_hip.get_device_count()
                    assert count > 0, "No devices in child"

                    ptr = child_hip.malloc(1024)
                    assert ptr != 0, "hipMalloc returned null in child"

                    child_hip.free(ptr)

                    print("CHILD_OK")
                    sys.stdout.flush()
                    os._exit(0)
                except Exception as e:
                    print(f"CHILD_FAIL: {e}", file=sys.stderr)
                    sys.stderr.flush()
                    os._exit(1)
            else:
                # Parent waits for child
                _, status = os.waitpid(pid, 0)
                exit_code = os.waitstatus_to_exitcode(status)
                assert exit_code == 0, f"Child exited with code {exit_code}"
                print("PARENT_OK: child fork+HIP succeeded")
        """
        result = _run_standalone(script, mem_limit="1G",
                                 extra_env={"TF_LOG_LEVEL": "hip_limiter=debug",
                                            "TF_LOG_PATH": "stderr"})
        assert result.returncode == 0, f"Fork safety test failed:\n{result.stderr}"
        assert "CHILD_OK" in result.stdout, \
            f"Child did not succeed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PARENT_OK" in result.stdout, \
            f"Parent did not confirm child success:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        # Verify fork safety came from sysfs (not lucky HIP tolerance)
        output = result.stdout + result.stderr
        assert "enumerated GPUs via KFD sysfs" in output, \
            f"Fork safety not via sysfs:\n{output}"

    def _run_with_debug_logs(self, mem_limit="1G"):
        """Run a minimal standalone script with debug logging to stderr."""
        script = """\
            from hip_helper import HIPRuntime
            hip = HIPRuntime()
            ptr = hip.malloc(1024)
            hip.free(ptr)
            print("PASS")
        """
        result = _run_standalone(script, mem_limit=mem_limit,
                                 extra_env={"TF_LOG_LEVEL": "hip_limiter=debug",
                                            "TF_LOG_PATH": "stderr"})
        assert result.returncode == 0, f"Failed: {result.stderr}"
        return result.stdout + result.stderr

    def test_sysfs_enumeration_used(self):
        """Verify standalone mode uses KFD sysfs enumeration (not HIP runtime)."""
        output = self._run_with_debug_logs()
        assert "enumerated GPUs via KFD sysfs" in output, \
            f"Expected sysfs enumeration log:\n{output}"

    def test_post_init_verification_runs(self):
        """Verify the one-time sysfs-vs-HIP verification runs and devices agree."""
        output = self._run_with_debug_logs()
        assert "DEVICE MISMATCH" not in output, \
            f"Sysfs/HIP device mismatch detected:\n{output}"
        assert "sysfs matches HIP device order" in output, \
            f"Expected verification log:\n{output}"


def test_reconciliation_reaps_dead_process_overhead():
    """Verify that reconciliation reaps stale non_hip overhead from dead processes.

    Scenario that caused 84% phantom overhead in CI (hnsnl pod):
    1. Process A starts, does 100+ allocs (triggers reconciliation → writes non_hip to proc slot)
    2. Process A is SIGKILLed (no drain → stale non_hip remains in proc slot)
    3. Process B starts, does 100+ allocs (triggers reconciliation → reap runs → clears stale slot)

    We verify "Reaped dead process" appears in process B's logs, confirming
    the reconciliation path triggers reap.
    """
    script_a = """\
        import signal, os
        from hip_helper import HIPRuntime
        hip = HIPRuntime()
        ptrs = []
        for _ in range(101):
            ptrs.append(hip.malloc(1024))
        # SIGKILL ourselves — no atexit drain, stale proc slot remains
        os.kill(os.getpid(), signal.SIGKILL)
    """
    script_b = """\
        from hip_helper import HIPRuntime
        hip = HIPRuntime()
        ptrs = []
        for _ in range(101):
            ptrs.append(hip.malloc(1024))
        for p in ptrs:
            hip.free(p)
        print("PASS")
    """
    with tempfile.TemporaryDirectory(prefix="cts_reap_") as tmpdir:
        shm_dir = os.path.join(tmpdir, "shm")
        os.makedirs(shm_dir, exist_ok=True)
        env = _standalone_env(shm_dir, mem_limit="1G",
                              extra_env={"TF_LOG_PATH": "stderr"})
        cts_dir = os.path.dirname(os.path.abspath(__file__))

        # Process A: allocate then SIGKILL (leaves stale proc slot)
        script_a_path = os.path.join(tmpdir, "script_a.py")
        with open(script_a_path, "w") as f:
            f.write(textwrap.dedent(script_a))
        proc_a = subprocess.run(
            [sys.executable, script_a_path],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT,
            env=env, cwd=cts_dir,
        )
        assert proc_a.returncode == -9, \
            f"Process A should have been SIGKILLed, got rc={proc_a.returncode}"

        # Process B: allocate (triggers reconciliation → reap)
        script_b_path = os.path.join(tmpdir, "script_b.py")
        with open(script_b_path, "w") as f:
            f.write(textwrap.dedent(script_b))
        proc_b = subprocess.run(
            [sys.executable, script_b_path],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT,
            env=env, cwd=cts_dir,
        )
        assert proc_b.returncode == 0, f"Process B failed: {proc_b.stderr}"
        output_b = proc_b.stdout + proc_b.stderr
        assert "Reaped dead process" in output_b, \
            f"Expected reap during reconciliation:\n{output_b}"


def test_high_overhead_warning():
    """Verify HIGH OVERHEAD warning fires when non-hipMalloc overhead > 25% of mem_limit.

    Strategy: set mem_limit very low (3M ≈ 2.8 MiB) so that the kernel's baseline
    VRAM overhead (~2 MiB for context + page tables) exceeds 25% of the limit.
    Then do 100 small allocs to trigger reconciliation, which measures DRM
    resident vs tracked and should log the warning.
    """
    script = """\
        from hip_helper import HIPRuntime
        hip = HIPRuntime()
        ptrs = []
        for _ in range(101):
            ptrs.append(hip.malloc(1024))
        for p in ptrs:
            hip.free(p)
        print("PASS")
    """
    result = _run_standalone(script, mem_limit="3M",
                             extra_env={"TF_LOG_PATH": "stderr"})
    assert result.returncode == 0, f"Failed: {result.stderr}"
    output = result.stdout + result.stderr
    assert "HIGH OVERHEAD" in output, \
        f"Expected HIGH OVERHEAD warning in logs:\n{output}"
