use std::sync::atomic::Ordering;
use utils::shared_memory::proc_slots::{ProcSlotTable, MAX_PROC_SLOTS};

/// Simulate the stale-PID scenario: process A allocates, "dies" (we just
/// leave its slot), process B inits and reaps A's usage via try_claim_and_release.
#[test]
fn reap_dead_pid_corrects_counter() {
    let table = ProcSlotTable::new_zeroed();
    table.initialize();

    // Process A claims slot, records 10 GB usage on device 0
    let slot_a = table.claim_slot(99999).unwrap(); // fake PID (dead)
    table.add_usage(slot_a, 0, 10 * 1024 * 1024 * 1024);

    // Simulate pod_memory_used = 10 GB
    let pod_memory_used = std::sync::atomic::AtomicU64::new(10 * 1024 * 1024 * 1024);

    // Process B finds dead PID 99999 and reaps using the two-step protocol
    let dead = table.find_dead_slots();
    assert_eq!(dead.len(), 1);
    let (dead_slot, dead_pid) = dead[0];
    assert_eq!(dead_pid, 99999);

    // Atomically claim and release — returns usage snapshot
    let usage = table.try_claim_and_release(dead_slot, dead_pid).unwrap();
    assert_eq!(usage[0], 10 * 1024 * 1024 * 1024);

    // Subtract from counter
    let prev = pod_memory_used.fetch_sub(usage[0], Ordering::AcqRel);
    assert_eq!(prev, 10 * 1024 * 1024 * 1024);
    assert_eq!(pod_memory_used.load(Ordering::Acquire), 0);

    // Slot is already released by try_claim_and_release — verify
    assert_eq!(table.slots[dead_slot].pid.load(Ordering::Relaxed), 0);
}

/// Concurrent slot claiming: multiple threads race to claim slots.
#[test]
fn concurrent_slot_claiming_no_duplicates() {
    let table = ProcSlotTable::new_zeroed();
    table.initialize();
    let table = Box::leak(table); // static lifetime for threads
    let table_ptr = table as *const ProcSlotTable as usize;

    let handles: Vec<_> = (0..MAX_PROC_SLOTS)
        .map(|i| {
            std::thread::spawn(move || {
                let table = unsafe { &*(table_ptr as *const ProcSlotTable) };
                table.claim_slot((i + 1) as u32)
            })
        })
        .collect();

    let mut claimed: Vec<usize> = Vec::new();
    for h in handles {
        if let Some(idx) = h.join().unwrap() {
            claimed.push(idx);
        }
    }

    // All slots should be claimed, no duplicates
    assert_eq!(claimed.len(), MAX_PROC_SLOTS);
    claimed.sort();
    claimed.dedup();
    assert_eq!(claimed.len(), MAX_PROC_SLOTS);

    // Cleanup: leak is intentional for test (process exit frees it)
}

/// SimulatedLimiter proc_usage invariant: after sequential ops,
/// proc_usage must equal pod_memory_used.
#[test]
fn simulated_limiter_proc_usage_equals_pod_used() {
    use hip_limiter_fuzz::SimulatedLimiter;

    let limiter = SimulatedLimiter::new(10_000_000);
    let mut live: Vec<usize> = Vec::new();

    // Alloc a bunch
    for _ in 0..50 {
        if let Ok(ptr) = limiter.try_alloc(100_000) {
            live.push(ptr);
        }
    }
    assert_eq!(limiter.proc_usage(), limiter.pod_memory_used());

    // Free half
    for _ in 0..25 {
        if let Some(ptr) = live.pop() {
            limiter.free(ptr);
        }
    }
    assert_eq!(limiter.proc_usage(), limiter.pod_memory_used());

    // Drain
    limiter.drain_allocations();
    assert_eq!(limiter.proc_usage(), 0);
    assert_eq!(limiter.pod_memory_used(), 0);
}

/// SimulatedLimiter pitched alloc updates proc_usage with actual_size.
#[test]
fn simulated_limiter_pitched_alloc_proc_usage() {
    use hip_limiter_fuzz::SimulatedLimiter;

    let limiter = SimulatedLimiter::new(10_000_000);

    // Pitched: estimated=1000, actual=1024 (alignment overhead)
    let ptr = limiter.try_alloc_pitched(1000, 1024, true).unwrap();
    assert_eq!(limiter.proc_usage(), 1024); // tracked at actual_size
    assert_eq!(limiter.pod_memory_used(), 1024);

    limiter.free(ptr);
    assert_eq!(limiter.proc_usage(), 0);
}

/// try_alloc_native_fails must NOT affect proc_usage.
#[test]
fn simulated_limiter_native_fail_no_proc_usage() {
    use hip_limiter_fuzz::SimulatedLimiter;

    let limiter = SimulatedLimiter::new(10_000_000);
    let _ = limiter.try_alloc_native_fails(5000);
    assert_eq!(limiter.proc_usage(), 0);
    assert_eq!(limiter.pod_memory_used(), 0);
}

/// Concurrent add_usage on same slot from multiple threads.
/// The final usage must equal the sum of all increments.
#[test]
fn concurrent_add_usage_same_slot() {
    use utils::shared_memory::MAX_DEVICES;

    let table = ProcSlotTable::new_zeroed();
    table.initialize();
    let slot = table.claim_slot(std::process::id()).unwrap();

    let table = Box::leak(table);
    let table_ptr = table as *const ProcSlotTable as usize;

    let thread_count = 8;
    let increments_per_thread: u64 = 1000;
    let increment_size: u64 = 100;

    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            std::thread::spawn(move || {
                let table = unsafe { &*(table_ptr as *const ProcSlotTable) };
                for _ in 0..increments_per_thread {
                    table.add_usage(slot, 0, increment_size);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let usage = table.read_slot_usage(slot);
    let expected = thread_count * increments_per_thread * increment_size;
    assert_eq!(usage[0], expected, "concurrent add_usage must sum correctly");

    // Other devices must be untouched
    for d in 1..MAX_DEVICES {
        assert_eq!(usage[d], 0, "device {d} must be 0");
    }
}

/// Concurrent try_claim_and_release: N threads race to reap the same dead slot.
/// Exactly one must win (get Some), all others must lose (get None).
/// The total usage subtracted must equal exactly the dead slot's usage.
#[test]
fn concurrent_try_claim_and_release_single_winner() {
    let table = ProcSlotTable::new_zeroed();
    table.initialize();

    let dead_pid: u32 = 4_000_000;
    let idx = table.claim_slot(dead_pid).unwrap();
    table.add_usage(idx, 0, 7777);
    table.add_usage(idx, 1, 3333);

    let table = Box::leak(table);
    let table_ptr = table as *const ProcSlotTable as usize;

    let thread_count = 16;
    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            std::thread::spawn(move || {
                let table = unsafe { &*(table_ptr as *const ProcSlotTable) };
                table.try_claim_and_release(idx, dead_pid)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let winners: Vec<_> = results.iter().filter_map(|r| r.as_ref()).collect();

    assert_eq!(winners.len(), 1, "exactly one thread must win the CAS");
    assert_eq!(winners[0][0], 7777);
    assert_eq!(winners[0][1], 3333);

    // Slot is now free
    assert_eq!(table.slots[idx].pid.load(Ordering::Relaxed), 0);
}

/// Concurrent reap_dead end-to-end: multiple threads call find_dead_slots +
/// try_claim_and_release on overlapping dead slots. Total reaped usage across
/// all threads must equal the sum of all dead slots' usage (no double-reap).
#[test]
fn concurrent_reap_dead_no_double_subtraction() {
    use std::sync::atomic::AtomicU64;

    let table = ProcSlotTable::new_zeroed();
    table.initialize();

    // Create 8 dead slots with known usage
    let dead_count = 8;
    let usage_per_slot: u64 = 1000;
    for i in 0..dead_count {
        let pid = 4_000_000 + i as u32;
        let idx = table.claim_slot(pid).unwrap();
        table.add_usage(idx, 0, usage_per_slot);
    }

    let table = Box::leak(table);
    let table_ptr = table as *const ProcSlotTable as usize;

    // Shared counter to accumulate reaped usage
    let total_reaped = Box::leak(Box::new(AtomicU64::new(0)));
    let reaped_ptr = total_reaped as *const AtomicU64 as usize;

    let thread_count = 16;
    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            std::thread::spawn(move || {
                let table = unsafe { &*(table_ptr as *const ProcSlotTable) };
                let reaped = unsafe { &*(reaped_ptr as *const AtomicU64) };

                let dead = table.find_dead_slots();
                for (slot_idx, dead_pid) in dead {
                    if let Some(usage) = table.try_claim_and_release(slot_idx, dead_pid) {
                        reaped.fetch_add(usage[0], Ordering::AcqRel);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let expected = dead_count as u64 * usage_per_slot;
    assert_eq!(
        total_reaped.load(Ordering::Acquire),
        expected,
        "total reaped usage must equal sum of all dead slots (no double-reap)"
    );

    // All dead slots should now be free
    for i in 0..dead_count {
        assert_eq!(
            table.slots[i].pid.load(Ordering::Relaxed),
            0,
            "slot {i} should be released"
        );
    }
}

/// Slot exhaustion: when all MAX_PROC_SLOTS are taken, claim_slot returns None
/// and the system degrades gracefully (no panic, no corruption).
#[test]
fn slot_exhaustion_graceful_degradation() {
    let table = ProcSlotTable::new_zeroed();
    table.initialize();

    // Fill all slots
    let mut slots = Vec::new();
    for i in 0..MAX_PROC_SLOTS {
        let idx = table.claim_slot((i + 1) as u32).unwrap();
        slots.push(idx);
    }

    // Next claim must fail
    assert!(table.claim_slot(9999).is_none(), "table should be full");

    // add_usage on valid slots still works
    table.add_usage(slots[0], 0, 1000);
    assert_eq!(table.read_slot_usage(slots[0])[0], 1000);

    // Release one slot, then claim succeeds
    table.zero_and_release(slots[0]);
    let new_slot = table.claim_slot(9999).unwrap();
    assert_eq!(new_slot, slots[0], "should reuse released slot");
    assert_eq!(table.read_slot_usage(new_slot)[0], 0, "counters must be zeroed");
}

/// TOCTOU defense: simulates the exact C1 race scenario where a dead slot is
/// reaped and reclaimed by a new live process between find_dead_slots and the
/// reap CAS. The fix passes the *original* PID to try_claim_and_release, so the
/// CAS correctly fails when the slot holds a new PID — preventing live-process eviction.
///
/// Sequence:
/// 1. Dead PID 4_000_000 occupies slot with 5000 bytes usage
/// 2. Reaper A calls find_dead_slots → sees (slot, 4_000_000)
/// 3. Reaper B reaps the slot (CAS 4_000_000 → 0) and zeros counters
/// 4. New process claims the slot with PID 42, writes 9999 bytes
/// 5. Reaper A calls try_claim_and_release(slot, 4_000_000) → must fail
///    (because the slot now holds PID 42, not 4_000_000)
///
/// Without the fix (if we re-read PID from the slot), reaper A would see PID 42,
/// CAS against 42, and evict the live process.
#[test]
fn toctou_reap_does_not_evict_new_occupant() {
    let table = ProcSlotTable::new_zeroed();
    table.initialize();

    let dead_pid: u32 = 4_000_000;
    let idx = table.claim_slot(dead_pid).unwrap();
    table.add_usage(idx, 0, 5000);

    // Step 2: Reaper A snapshots dead slots (would happen in find_dead_slots)
    let reaper_a_snapshot = table.find_dead_slots();
    assert_eq!(reaper_a_snapshot.len(), 1);
    let (snapshot_slot, snapshot_pid) = reaper_a_snapshot[0];
    assert_eq!(snapshot_pid, dead_pid);

    // Step 3: Reaper B swoops in first and reaps
    let reaper_b_result = table.try_claim_and_release(idx, dead_pid);
    assert!(reaper_b_result.is_some());
    assert_eq!(reaper_b_result.unwrap()[0], 5000);

    // Step 4: New live process claims the now-free slot
    let new_pid: u32 = 42;
    let new_idx = table.claim_slot(new_pid).unwrap();
    assert_eq!(new_idx, idx, "should reuse the same slot");
    table.add_usage(new_idx, 0, 9999);

    // Step 5: Reaper A tries to reap using its stale snapshot — must fail
    let reaper_a_result = table.try_claim_and_release(snapshot_slot, snapshot_pid);
    assert!(
        reaper_a_result.is_none(),
        "CAS must fail: slot now holds PID {new_pid}, not {dead_pid}"
    );

    // Verify the live process is untouched
    assert_eq!(
        table.slots[idx].pid.load(Ordering::Relaxed),
        new_pid,
        "live process must still own the slot"
    );
    assert_eq!(
        table.read_slot_usage(idx)[0],
        9999,
        "live process usage must be intact"
    );
}

/// TOCTOU defense: simulates the C2 race scenario where usage is read AFTER
/// the CAS releases the slot, and a new claimer zeroes the counters first.
///
/// The fix reads usage BEFORE the CAS, so we capture the dead process's final
/// values before the slot becomes available for reclaim.
///
/// Sequence:
/// 1. Dead PID occupies slot with 7777 bytes on device 0
/// 2. Reaper reads usage snapshot (7777) then CAS succeeds (slot freed)
/// 3. New process immediately claims slot and zeroes counters
/// 4. The reaper's usage snapshot must still be 7777 (not 0)
#[test]
fn toctou_usage_snapshot_before_cas_not_after() {
    let table = ProcSlotTable::new_zeroed();
    table.initialize();

    let dead_pid: u32 = 4_000_000;
    let idx = table.claim_slot(dead_pid).unwrap();
    table.add_usage(idx, 0, 7777);

    // try_claim_and_release reads usage BEFORE CAS, then CAS frees the slot
    let usage = table.try_claim_and_release(idx, dead_pid);
    assert!(usage.is_some());

    // Immediately after the CAS, a new process claims the slot (zeroes counters)
    let new_idx = table.claim_slot(42).unwrap();
    assert_eq!(new_idx, idx);

    // The usage snapshot from the reaper must reflect the dead process's data,
    // NOT the zeroed counters from the new claimer
    assert_eq!(
        usage.unwrap()[0],
        7777,
        "usage snapshot must be from before CAS (dead process data), not after (zeroed)"
    );
}

/// Reap-on-OOM: stale usage blocks allocation, reap recovers it, retry succeeds.
#[test]
fn reap_on_oom_recovers_stale_usage() {
    use hip_limiter_fuzz::SimulatedLimiter;

    let limiter = SimulatedLimiter::new(1_000_000);

    // Fill to 900K with real allocations
    let ptr = limiter.try_alloc(900_000).unwrap();
    assert_eq!(limiter.pod_memory_used(), 900_000);

    // Simulate dead process that left 200K stale usage in the counter
    limiter.inject_stale_usage(200_000);
    assert_eq!(limiter.pod_memory_used(), 1_100_000);

    // Direct alloc of 50K should fail (1_100_000 + 50_000 > 1_000_000)
    assert!(limiter.try_alloc(50_000).is_err());

    // With reap recovering 200K, retry succeeds
    let ptr2 = limiter.try_alloc_with_reap(50_000, |l| {
        l.recover_stale_usage(200_000);
        1 // reaped 1 dead process
    });
    assert!(ptr2.is_ok());
    // 900K (real) + 50K (new) = 950K
    assert_eq!(limiter.pod_memory_used(), 950_000);

    limiter.free(ptr);
    limiter.free(ptr2.unwrap());
    assert_eq!(limiter.pod_memory_used(), 0);
}

/// Reap-on-OOM: reap doesn't help enough — retry still fails.
#[test]
fn reap_on_oom_insufficient_recovery() {
    use hip_limiter_fuzz::SimulatedLimiter;

    let limiter = SimulatedLimiter::new(1_000_000);

    // Fill to 950K
    let _ptr = limiter.try_alloc(950_000).unwrap();

    // 100K stale usage (total = 1_050_000)
    limiter.inject_stale_usage(100_000);

    // Try to alloc 200K — even after reaping 100K, 950K + 200K > 1M
    let result = limiter.try_alloc_with_reap(200_000, |l| {
        l.recover_stale_usage(100_000);
        1
    });
    assert!(result.is_err());
    // pod_memory_used should be back to 950K (stale was reaped, but alloc was denied)
    assert_eq!(limiter.pod_memory_used(), 950_000);
}

/// Reap-on-OOM: no stale usage to reap — fails immediately without retry.
#[test]
fn reap_on_oom_nothing_to_reap() {
    use hip_limiter_fuzz::SimulatedLimiter;

    let limiter = SimulatedLimiter::new(1_000_000);
    let _ptr = limiter.try_alloc(900_000).unwrap();

    let mut reap_called = false;
    let result = limiter.try_alloc_with_reap(200_000, |_l| {
        reap_called = true;
        0 // nothing to reap
    });
    assert!(result.is_err());
    assert!(reap_called, "reap_fn must be called on OOM");
    assert_eq!(limiter.pod_memory_used(), 900_000, "no side effects from failed reap path");
}
