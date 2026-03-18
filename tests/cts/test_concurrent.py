"""
CTS Concurrent Stress Tests — Multi-thread and multi-process allocation stress.

Validates that the hip-limiter correctly tracks memory accounting under concurrent
access patterns. The SHM pod_memory_used field uses atomic operations, so it must
remain consistent even when multiple threads or processes allocate/free simultaneously.

Key invariant: after all threads/processes finish and all live allocations are known,
SHM pod_memory_used must equal the sum of sizes of allocations that are still live.
"""

import concurrent.futures

import pytest

from conftest import DEFAULT_TEST_UUID, MiB, GiB, parse_kv_output, requires_gpu
from shm_writer import DeviceSpec

pytestmark = requires_gpu

ALLOC_SIZE = 1 * MiB  # 1 MiB per allocation — small enough to fit many within limit
NUM_THREADS = 8
ALLOCS_PER_THREAD = 10


class TestMultithreadAlloc:
    """N threads allocating simultaneously; verify total tracked == sum of successful allocs."""

    def test_multithread_alloc(self, cts):
        """Spawn N threads that each allocate ALLOCS_PER_THREAD buffers, then verify
        that SHM pod_memory_used matches the total successfully allocated."""
        script = f"""\
import os
import threading
from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY
from shm_writer import read_pod_memory_used

NUM_THREADS = {NUM_THREADS}
ALLOCS_PER_THREAD = {ALLOCS_PER_THREAD}
ALLOC_SIZE = {ALLOC_SIZE}

hip = HIPRuntime()
results = {{}}
lock = threading.Lock()

def alloc_worker(thread_id):
    success_count = 0
    pointers = []
    for _ in range(ALLOCS_PER_THREAD):
        err, ptr = hip.malloc_raw(ALLOC_SIZE)
        if err == HIP_SUCCESS and ptr != 0:
            success_count += 1
            pointers.append(ptr)
    with lock:
        results[thread_id] = {{"success": success_count, "ptrs": pointers}}

threads = []
for tid in range(NUM_THREADS):
    t = threading.Thread(target=alloc_worker, args=(tid,))
    threads.append(t)

for t in threads:
    t.start()
for t in threads:
    t.join()

total_success = sum(r["success"] for r in results.values())
print(f"TOTAL_SUCCESS={{total_success}}")
print(f"EXPECTED_BYTES={{total_success * ALLOC_SIZE}}")
shm_used = read_pod_memory_used(os.environ["TF_SHM_FILE"], 0)
print(f"SHM_USED={{shm_used}}")
"""
        result = cts.run_hip_test(script, timeout=60)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"

        # Parse output
        values = parse_kv_output(result.stdout)

        total_success = values.get("TOTAL_SUCCESS", 0)
        expected_bytes = values.get("EXPECTED_BYTES", 0)
        assert total_success > 0, "No allocations succeeded — is the GPU available?"

        # SHM pod_memory_used reported by subprocess (before atexit cleanup)
        shm_used = values.get("SHM_USED", 0)
        assert shm_used == expected_bytes, (
            f"SHM pod_memory_used ({shm_used}) != expected ({expected_bytes})"
        )


class TestMultiprocessAlloc:
    """N processes sharing the same SHM file; verify combined accounting.

    Validates that concurrent subprocesses can safely update SHM pod_memory_used
    via atomic fetch_add. Assumes all subprocesses exit cleanly — if a subprocess
    crashed after fetch_add but before printing ALLOC_OK, SHM would be higher
    than expected (the known Option A over-reporting tradeoff).
    """

    def test_multiprocess_alloc(self, cts):
        """Spawn N subprocesses each making a single allocation without freeing.
        Each process's atexit handler drains its tracked allocations, so after
        all processes exit, SHM pod_memory_used should return to 0."""
        num_procs = 4
        alloc_size = 2 * MiB

        # Each subprocess allocates one buffer, reports SHM mid-flight, then exits.
        # The atexit handler decrements pod_memory_used for the unfreed allocation.
        child_script = f"""\
import os
from hip_helper import HIPRuntime, HIP_SUCCESS
from shm_writer import read_pod_memory_used
hip = HIPRuntime()
err, ptr = hip.malloc_raw({alloc_size})
if err == HIP_SUCCESS and ptr != 0:
    shm_used = read_pod_memory_used(os.environ["TF_SHM_FILE"], 0)
    print(f"ALLOC_OK SHM_USED={{shm_used}}")
else:
    print(f"ALLOC_FAIL={{err}}")
# Do NOT free — atexit handler will drain allocation tracker
"""
        # Spawn all subprocesses concurrently rather than sequentially
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_procs) as executor:
            futures = [executor.submit(cts.run_hip_test, child_script) for _ in range(num_procs)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result.succeeded and "ALLOC_OK" in result.stdout:
                    success_count += 1

        assert success_count > 0, (
            "No subprocess allocations succeeded — test would pass trivially"
        )

        # After all processes exit, their atexit handlers should have drained
        # all tracked allocations. SHM pod_memory_used should be 0.
        shm_used = cts.read_pod_memory_used(device_idx=0)
        assert shm_used == 0, (
            f"SHM pod_memory_used ({shm_used}) should be 0 after all {success_count} "
            f"processes exited (atexit handler drains tracked allocations)"
        )


class TestMultiprocessAllocFreeZero:
    """N processes each allocate and free; verify SHM returns to 0.

    Gap: TestMultiprocessAlloc only tests the leak path (alloc without free).
    This test validates the non-leak happy path: multiple independent processes
    each allocate, free, and exit cleanly. After all processes complete, SHM
    pod_memory_used must be exactly 0 — proving that cross-process atomic
    fetch_sub on free correctly decrements the shared counter.
    """

    def test_multiprocess_alloc_free_returns_to_zero(self, cts):
        num_procs = 4
        alloc_size = 2 * MiB

        child_script = f"""\
from hip_helper import HIPRuntime, HIP_SUCCESS
hip = HIPRuntime()
err, ptr = hip.malloc_raw({alloc_size})
if err != HIP_SUCCESS or ptr == 0:
    print(f"ALLOC_FAIL={{err}}")
else:
    free_err = hip.free_raw(ptr)
    if free_err == HIP_SUCCESS:
        print("FREE_OK")
    else:
        print(f"FREE_FAIL={{free_err}}")
"""
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_procs) as executor:
            futures = [executor.submit(cts.run_hip_test, child_script) for _ in range(num_procs)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result.succeeded and "FREE_OK" in result.stdout:
                    success_count += 1

        assert success_count > 0, (
            "No subprocess alloc+free cycles succeeded — test would pass trivially"
        )

        shm_used = cts.read_pod_memory_used(device_idx=0)
        assert shm_used == 0, (
            f"SHM pod_memory_used ({shm_used}) should be 0 after all {success_count} "
            f"processes allocated and freed {alloc_size} bytes each"
        )


class TestAllocFreeInterleaved:
    """Threads interleaving allocs and frees; verify final SHM consistency.

    Validates that concurrent fetch_add (alloc) and saturating_fetch_sub (free)
    on the same pod_memory_used atomic don't drift. 20 cycles per thread is
    enough to create substantial interleaving without making the test slow.
    """

    def test_alloc_free_interleaved(self, cts):
        """Each thread allocates a buffer, frees it, and repeats. After all threads
        join, all memory should be freed and SHM pod_memory_used should be 0."""
        script = f"""\
import threading
from hip_helper import HIPRuntime, HIP_SUCCESS

NUM_THREADS = {NUM_THREADS}
CYCLES = 20
ALLOC_SIZE = {ALLOC_SIZE}

hip = HIPRuntime()
errors = []
lock = threading.Lock()

def worker(thread_id):
    for _ in range(CYCLES):
        err, ptr = hip.malloc_raw(ALLOC_SIZE)
        if err != HIP_SUCCESS or ptr == 0:
            with lock:
                errors.append(f"thread {{thread_id}}: alloc failed (err={{err}})")
            return
        free_err = hip.free_raw(ptr)
        if free_err != HIP_SUCCESS:
            with lock:
                errors.append(f"thread {{thread_id}}: free failed (err={{free_err}})")
            return

threads = []
for tid in range(NUM_THREADS):
    t = threading.Thread(target=worker, args=(tid,))
    threads.append(t)

for t in threads:
    t.start()
for t in threads:
    t.join()

if errors:
    print(f"ERRORS={{len(errors)}}")
    for e in errors[:5]:
        print(f"  {{e}}")
else:
    print("ALL_OK")
"""
        result = cts.run_hip_test(script, timeout=60)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "ALL_OK" in result.stdout, f"Worker errors:\n{result.stdout}"

        # All allocs were freed, so SHM should show 0 usage
        shm_used = cts.read_pod_memory_used(device_idx=0)
        assert shm_used == 0, (
            f"SHM pod_memory_used ({shm_used}) should be 0 after all allocs freed"
        )


class TestMultithreadWithLimit:
    """Concurrent allocations that collectively exceed the memory limit."""

    def test_multithread_exceeds_limit(self, cts_factory):
        """With a tight limit, many threads compete. Total tracked should never
        exceed the limit (some allocs will be denied with hipErrorOutOfMemory)."""
        mem_limit = 16 * MiB
        fixture = cts_factory(
            devices=[DeviceSpec(uuid=DEFAULT_TEST_UUID, mem_limit=mem_limit, device_idx=0)]
        )
        alloc_size = 2 * MiB

        script = f"""\
import os
import threading
from hip_helper import HIPRuntime, HIP_SUCCESS, HIP_ERROR_OUT_OF_MEMORY
from shm_writer import read_pod_memory_used

NUM_THREADS = 16
ALLOC_SIZE = {alloc_size}

hip = HIPRuntime()
results = {{"success": 0, "denied": 0, "other_err": 0}}
lock = threading.Lock()

def worker():
    err, ptr = hip.malloc_raw(ALLOC_SIZE)
    with lock:
        if err == HIP_SUCCESS and ptr != 0:
            results["success"] += 1
        elif err == HIP_ERROR_OUT_OF_MEMORY:
            results["denied"] += 1
        else:
            results["other_err"] += 1

threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"SUCCESS={{results['success']}}")
print(f"DENIED={{results['denied']}}")
print(f"OTHER_ERR={{results['other_err']}}")
shm_used = read_pod_memory_used(os.environ["TF_SHM_FILE"], 0)
print(f"SHM_USED={{shm_used}}")
"""
        result = fixture.run_hip_test(script, timeout=60)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"

        values = parse_kv_output(result.stdout)

        success_count = values.get("SUCCESS", 0)
        denied_count = values.get("DENIED", 0)

        # The limiter uses reserve-then-allocate (atomic fetch_add before limit check),
        # so pod_memory_used can never exceed mem_limit. At most mem_limit / alloc_size
        # allocations can succeed, regardless of concurrency.
        max_possible = mem_limit // alloc_size
        assert success_count <= max_possible, (
            f"More allocs succeeded ({success_count}) than limit allows ({max_possible})"
        )
        # With 16 threads competing for 8 slots, at least some must be denied.
        assert denied_count > 0, (
            f"Expected some allocations to be denied, "
            f"but got success={success_count}, denied={denied_count}"
        )

        # SHM usage reported by subprocess (before atexit cleanup)
        shm_used = values.get("SHM_USED", 0)
        expected = success_count * alloc_size
        assert shm_used == expected, (
            f"SHM pod_memory_used ({shm_used}) != expected ({expected})"
        )
