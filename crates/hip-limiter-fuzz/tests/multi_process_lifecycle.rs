use proptest::prelude::*;

use hip_limiter_fuzz::SimulatedPod;

const GIB: u64 = 1024 * 1024 * 1024;

/// Stale non-hipMalloc overhead from killed processes is cleared by reconciliation.
/// Models CI scenario: multiple training processes crash, one survivor reconciles.
#[test]
fn stale_non_hip_cleared_by_reconciliation() {
    let mut pod = SimulatedPod::new(&[128 * GIB]);

    // Spawn 4 processes each reporting 13 GiB non-hip overhead.
    let procs: Vec<usize> = (0..4).map(|_| pod.spawn_process(&[13 * GIB])).collect();

    // Kill 3 of them — their slots become stale.
    for &p in &procs[..3] {
        pod.kill_process(p);
    }

    // Before reconciliation: all 4 slots still report non_hip.
    assert_eq!(pod.sum_non_hip_for_device(0), 4 * 13 * GIB);

    // Surviving process reconciles.
    let survivor = procs[3];
    pod.reconcile(survivor);

    // After reconciliation: only the live process's overhead remains.
    assert_eq!(pod.sum_non_hip_for_device(0), 13 * GIB);
    assert_eq!(pod.effective_mem_limit(0), 128 * GIB - 13 * GIB);
}

/// Rapid process cycling: spawn, alloc enough to trigger reconciliation, kill.
/// After 20 cycles, stale overhead must converge to zero (plus the fresh process).
#[test]
fn rapid_process_cycling_overhead_convergence() {
    let mut pod = SimulatedPod::new(&[128 * GIB]);

    for _ in 0..20 {
        let p = pod.spawn_process(&[5 * GIB]);

        // Do 101 allocations (reconciliation triggers at count 0 and 100).
        for _ in 0..101 {
            let _ = pod.try_alloc(p, 0, 1024);
        }

        pod.kill_process(p);
    }

    // All 20 processes are dead with stale slots.
    // Spawn a fresh process and reconcile — should reap everything.
    let fresh_non_hip = 2 * GIB;
    let fresh = pod.spawn_process(&[fresh_non_hip]);
    pod.reconcile(fresh);

    // Only the fresh process's overhead should remain.
    assert_eq!(pod.sum_non_hip_for_device(0), fresh_non_hip);

    // Effective limit should be mem_limit minus fresh process's overhead.
    assert_eq!(pod.effective_mem_limit(0), 128 * GIB - fresh_non_hip);
}

/// Slot exhaustion: fill all 128 slots, kill them, spawn new process.
/// After reconciliation the dead slots are reaped and the new process works normally.
#[test]
fn slot_exhaustion_with_non_hip() {
    let mut pod = SimulatedPod::new(&[128 * GIB]);

    // Fill all 128 slots then kill the occupants — but leave 1 slot free
    // so we can spawn a live process that can reconcile.
    let dead_procs: Vec<usize> = (0..127).map(|_| pod.spawn_process(&[1 * GIB])).collect();
    for &p in &dead_procs {
        pod.kill_process(p);
    }

    // Spawn a live process — should claim one of the remaining slots.
    let live = pod.spawn_process(&[1 * GIB]);
    assert!(
        pod.process(live).slot_idx.is_some(),
        "should claim the last available slot"
    );

    // Before reconciliation: 127 dead + 1 live = 128 GiB of non_hip reported.
    assert_eq!(pod.sum_non_hip_for_device(0), 128 * GIB);

    // Reconcile — reaps the 127 dead slots.
    pod.reconcile(live);

    // After reap: only the live process's 1 GiB remains.
    assert_eq!(pod.sum_non_hip_for_device(0), 1 * GIB);
    assert_eq!(pod.effective_mem_limit(0), 127 * GIB);
}

/// Multiple surviving processes all reconcile — dead slots are reaped exactly once.
/// No double-counting of reaped overhead.
#[test]
fn concurrent_reconciliation_no_double_reap() {
    let mut pod = SimulatedPod::new(&[128 * GIB]);

    // Spawn 8 processes, each with 5 GiB non-hip overhead.
    let procs: Vec<usize> = (0..8).map(|_| pod.spawn_process(&[5 * GIB])).collect();

    // Each process does a small allocation.
    for &p in &procs {
        let _ = pod.try_alloc(p, 0, 1024).unwrap();
    }

    // Kill 4 of them.
    for &p in &procs[..4] {
        pod.kill_process(p);
    }

    // All 4 survivors reconcile in sequence.
    for &p in &procs[4..] {
        pod.reconcile(p);
    }

    // Dead processes' overhead is fully reaped (no double-reap).
    assert_eq!(pod.sum_non_hip_for_device(0), 4 * 5 * GIB);
    assert_eq!(pod.effective_mem_limit(0), 128 * GIB - 20 * GIB);

    // pod_memory_used should reflect only the 4 live processes' allocations.
    // Dead processes' 1024-byte allocs were reaped during reconciliation.
    assert_eq!(pod.pod_memory_used(0), 4 * 1024);
}

/// Effective limit tightens as more processes with overhead join the pod.
#[test]
fn effective_limit_tightens_with_more_processes() {
    let mut pod = SimulatedPod::new(&[128 * GIB]);

    for i in 1..=8u64 {
        let p = pod.spawn_process(&[5 * GIB]);
        pod.reconcile(p);
        assert_eq!(
            pod.effective_mem_limit(0),
            128 * GIB - i * 5 * GIB,
            "after spawning process {i}"
        );
    }
}

/// Killing processes and spawning new ones reuses reaped slots.
#[test]
fn kill_then_spawn_reuses_slot() {
    let mut pod = SimulatedPod::new(&[128 * GIB]);

    // Spawn 4 processes with 10 GiB non-hip each.
    let procs: Vec<usize> = (0..4).map(|_| pod.spawn_process(&[10 * GIB])).collect();

    // Kill 2 of them.
    pod.kill_process(procs[0]);
    pod.kill_process(procs[1]);

    // Spawn 2 new processes with 3 GiB non-hip each.
    let new0 = pod.spawn_process(&[3 * GIB]);
    let new1 = pod.spawn_process(&[3 * GIB]);

    // New processes should have valid slots.
    assert!(pod.process(new0).slot_idx.is_some());
    assert!(pod.process(new1).slot_idx.is_some());

    // Reconcile from one of the new processes — reaps dead slots.
    pod.reconcile(new0);

    // Live processes: procs[2] (10 GiB), procs[3] (10 GiB), new0 (3 GiB), new1 (3 GiB)
    assert_eq!(pod.sum_non_hip_for_device(0), 2 * 10 * GIB + 2 * 3 * GIB);
    assert_eq!(
        pod.effective_mem_limit(0),
        128 * GIB - (20 * GIB + 6 * GIB)
    );
}

/// Per-device overhead is independent — overhead on one GPU does not affect another.
#[test]
fn multi_device_overhead_independence() {
    let mut pod = SimulatedPod::new(&[128 * GIB, 128 * GIB, 128 * GIB, 128 * GIB]);

    // Spawn a process with varying overhead per device.
    let p = pod.spawn_process(&[50 * GIB, 10 * GIB, 0, 30 * GIB]);
    pod.reconcile(p);

    // Verify each device's effective limit independently.
    assert_eq!(pod.effective_mem_limit(0), 78 * GIB);
    assert_eq!(pod.effective_mem_limit(1), 118 * GIB);
    assert_eq!(pod.effective_mem_limit(2), 128 * GIB);
    assert_eq!(pod.effective_mem_limit(3), 98 * GIB);

    // Large alloc on device 2 (no overhead) succeeds.
    assert!(
        pod.try_alloc(p, 2, 100 * GIB).is_ok(),
        "device 2 has no overhead, 100 GiB alloc should succeed"
    );

    // Same alloc on device 0 (50 GiB overhead, effective=78 GiB) fails
    // because 100 > 78.
    assert!(
        pod.try_alloc(p, 0, 100 * GIB).is_err(),
        "device 0 has 50 GiB overhead, 100 GiB alloc should be denied"
    );
}

/// Clean drain releases the proc slot, so a subsequent reconciliation by another
/// process should see reduced total overhead and a looser effective limit.
#[test]
fn drain_updates_effective_limit_for_other_processes() {
    let mut pod = SimulatedPod::new(&[128 * GIB]);

    let p0 = pod.spawn_process(&[20 * GIB]);
    let p1 = pod.spawn_process(&[10 * GIB]);
    pod.reconcile(p0);

    // Both processes' overhead contributes: effective = 128 - 30 = 98 GiB.
    assert_eq!(pod.effective_mem_limit(0), 98 * GIB);

    // p0 drains cleanly (atexit) — releases its proc slot, zeroes non_hip.
    pod.drain_process(p0);

    // p1 reconciles — should see only its own 10 GiB overhead.
    pod.reconcile(p1);
    assert_eq!(pod.effective_mem_limit(0), 118 * GIB);
    assert_eq!(pod.sum_non_hip_for_device(0), 10 * GIB);
}

/// OOM recovery via reap in SimulatedPod: alloc denied → reap dead slots → retry succeeds.
#[test]
fn oom_reap_retry_succeeds() {
    let mut pod = SimulatedPod::new(&[20 * GIB]);

    // Process with 15 GiB overhead → effective = 5 GiB.
    let dead = pod.spawn_process(&[15 * GIB]);
    // Do 101 allocs to trigger reconciliation that sets effective limit.
    for _ in 0..101 {
        let _ = pod.try_alloc(dead, 0, 1024);
    }
    pod.kill_process(dead);

    // New process with 1 GiB overhead.
    let live = pod.spawn_process(&[1 * GIB]);

    // Before reap: effective limit is still mem_limit - (15 + 1) = 4 GiB.
    // Try to alloc 6 GiB — should fail because effective = 4 GiB.
    assert!(pod.try_alloc(live, 0, 6 * GIB).is_err());

    // Reconcile (triggers reap of dead process).
    pod.reconcile(live);

    // After reap: effective = 20 - 1 = 19 GiB. 6 GiB should now succeed.
    assert_eq!(pod.effective_mem_limit(0), 19 * GIB);
    assert!(pod.try_alloc(live, 0, 6 * GIB).is_ok());
}

/// Process with no proc slot (slot_idx == None) can still allocate and free,
/// but reconciliation doesn't corrupt effective_mem_limit.
#[test]
fn no_slot_process_does_not_corrupt_effective_limit() {
    let mut pod = SimulatedPod::new(&[128 * GIB]);

    // Fill all 128 slots.
    let holders: Vec<usize> = (0..128).map(|_| pod.spawn_process(&[0])).collect();

    // 129th process gets no slot.
    let no_slot = pod.spawn_process(&[5 * GIB]);
    assert!(
        pod.process(no_slot).slot_idx.is_none(),
        "should have no slot when all 128 are taken"
    );

    // Process with no slot can still allocate.
    let ptr = pod.try_alloc(no_slot, 0, 1024).unwrap();
    assert_eq!(pod.pod_memory_used(0), 1024);

    // Reconcile from no-slot process — should not panic or corrupt limits.
    pod.reconcile(no_slot);

    // Effective limit should be 128 GiB (no non_hip from any process since all have 0).
    assert_eq!(pod.effective_mem_limit(0), 128 * GIB);

    // Free still works.
    assert!(pod.free(no_slot, ptr));
    assert_eq!(pod.pod_memory_used(0), 0);

    // Clean up — release some slots so we can verify no corruption.
    pod.drain_process(holders[0]);
}

// --- Proptest: random spawn/kill/alloc/free/reconcile sequences ---

/// Actions that can be performed on a SimulatedPod.
#[derive(Debug, Clone)]
enum PodAction {
    /// Spawn a new process with given non_hip (in GiB, capped to reasonable range).
    Spawn(u8),
    /// Kill process at index (modulo live count).
    Kill(usize),
    /// Alloc on process (idx % live) on device 0.
    Alloc(usize, u32),
    /// Free on process (idx % live), pointer index (mod tracked).
    Free(usize, usize),
    /// Reconcile from process (idx % live).
    Reconcile(usize),
}

fn pod_action_strategy() -> impl Strategy<Value = PodAction> {
    prop_oneof![
        (0u8..20).prop_map(PodAction::Spawn),
        (0usize..100).prop_map(PodAction::Kill),
        (0usize..100, 1u32..10_000).prop_map(|(p, s)| PodAction::Alloc(p, s)),
        (0usize..100, 0usize..500).prop_map(|(p, ptr)| PodAction::Free(p, ptr)),
        (0usize..100).prop_map(PodAction::Reconcile),
    ]
}

proptest! {
    /// Random sequences of pod operations must maintain accounting invariants:
    /// 1. pod_memory_used is always <= effective_mem_limit (or mem_limit if effective==0)
    /// 2. effective_mem_limit is always <= mem_limit
    /// 3. After draining all live processes, pod_memory_used == 0
    #[test]
    fn pod_random_lifecycle_invariants(
        actions in proptest::collection::vec(pod_action_strategy(), 10..100),
    ) {
        let mem_limit = 100 * GIB;
        let mut pod = SimulatedPod::new(&[mem_limit]);
        let mut live_indices: Vec<usize> = Vec::new();
        // Per-process pointer tracking for Free actions.
        let mut proc_pointers: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();

        // Always start with one process so we have something to work with.
        let p0 = pod.spawn_process(&[GIB]);
        live_indices.push(p0);
        proc_pointers.insert(p0, Vec::new());

        for action in &actions {
            match action {
                PodAction::Spawn(non_hip_gib) => {
                    if live_indices.len() < 64 {
                        let p = pod.spawn_process(&[(*non_hip_gib as u64) * GIB]);
                        live_indices.push(p);
                        proc_pointers.insert(p, Vec::new());
                    }
                }
                PodAction::Kill(idx) => {
                    if live_indices.len() > 1 {
                        let i = *idx % live_indices.len();
                        let p = live_indices.remove(i);
                        proc_pointers.remove(&p);
                        pod.kill_process(p);
                    }
                }
                PodAction::Alloc(idx, size) => {
                    if !live_indices.is_empty() {
                        let p = live_indices[*idx % live_indices.len()];
                        if let Ok(ptr) = pod.try_alloc(p, 0, *size as u64 * 1024) {
                            proc_pointers.entry(p).or_default().push(ptr);
                        }
                    }
                }
                PodAction::Free(idx, ptr_idx) => {
                    if !live_indices.is_empty() {
                        let p = live_indices[*idx % live_indices.len()];
                        if let Some(ptrs) = proc_pointers.get_mut(&p) {
                            if !ptrs.is_empty() {
                                let i = *ptr_idx % ptrs.len();
                                let ptr = ptrs.swap_remove(i);
                                pod.free(p, ptr);
                            }
                        }
                    }
                }
                PodAction::Reconcile(idx) => {
                    if !live_indices.is_empty() {
                        let p = live_indices[*idx % live_indices.len()];
                        pod.reconcile(p);
                    }
                }
            }

            // Invariant: effective_mem_limit <= mem_limit (when non-zero).
            let effective = pod.effective_mem_limit(0);
            if effective > 0 {
                prop_assert!(effective <= mem_limit,
                    "effective {} > mem_limit {}", effective, mem_limit);
            }

            // Invariant: pod_memory_used <= active limit.
            let active = if effective > 0 { effective } else { mem_limit };
            prop_assert!(pod.pod_memory_used(0) <= active,
                "pod_memory_used {} > active_limit {}",
                pod.pod_memory_used(0), active);
        }

        // Final invariant: reconcile (to reap killed processes) then drain all
        // live processes → pod_memory_used == 0.
        if !live_indices.is_empty() {
            pod.reconcile(live_indices[0]);
        }
        for p in live_indices {
            pod.drain_process(p);
        }
        prop_assert_eq!(pod.pod_memory_used(0), 0,
            "pod_memory_used must be 0 after reap + drain of all processes");
    }
}
