use proptest::prelude::*;

use hip_limiter_fuzz::{MultiDeviceSimulatedLimiter, SimulatedLimiter};

#[derive(Debug, Clone)]
enum Operation {
    Alloc(u64),
    /// The value is an index into the live_pointers vec, NOT a pointer address.
    /// Modulo is used to map it to a valid index at runtime.
    Free(usize),
}

fn operation_strategy() -> impl Strategy<Value = Operation> {
    prop_oneof![
        (1u64..1_000_000).prop_map(Operation::Alloc),
        (0usize..500).prop_map(Operation::Free),
    ]
}

proptest! {
    /// After executing a random sequence of alloc/free operations,
    /// `pod_memory_used` must equal the sum of all live allocation sizes.
    #[test]
    fn alloc_free_accounting(operations in proptest::collection::vec(operation_strategy(), 1..500)) {
        let limiter = SimulatedLimiter::new(10_000_000);
        let mut live_pointers: Vec<usize> = Vec::new();

        for operation in &operations {
            match operation {
                Operation::Alloc(size) => {
                    if let Ok(pointer) = limiter.try_alloc(*size) {
                        if *size > 0 {
                            live_pointers.push(pointer);
                        }
                    }
                }
                Operation::Free(index) => {
                    if !live_pointers.is_empty() {
                        let idx = *index % live_pointers.len();
                        let pointer = live_pointers.swap_remove(idx);
                        limiter.free(pointer);
                    }
                }
            }
        }

        prop_assert_eq!(
            limiter.allocation_count(),
            live_pointers.len(),
            "allocation_count must match number of live pointers"
        );
        prop_assert_eq!(limiter.pod_memory_used(), limiter.tracked_total());
        prop_assert_eq!(limiter.proc_usage(), limiter.pod_memory_used(),
            "proc_usage must equal pod_memory_used in single-threaded scenario");
    }

    /// pod_memory_used must never exceed mem_limit in single-threaded usage
    /// (no TOCTOU race possible with a single thread).
    #[test]
    fn never_exceeds_limit(sizes in proptest::collection::vec(1u64..1_000_000, 1..100)) {
        let limit = 5_000_000u64;
        let limiter = SimulatedLimiter::new(limit);

        for size in &sizes {
            let _ = limiter.try_alloc(*size);
            prop_assert!(
                limiter.pod_memory_used() <= limit,
                "pod_memory_used ({}) exceeded limit ({})",
                limiter.pod_memory_used(),
                limit
            );
            prop_assert_eq!(limiter.proc_usage(), limiter.pod_memory_used(),
                "proc_usage must track pod_memory_used");
        }
    }

    /// Allocate everything, then free everything: pod_memory_used must return to zero.
    #[test]
    fn free_returns_to_zero(sizes in proptest::collection::vec(1u64..1_000_000, 1..50)) {
        let limiter = SimulatedLimiter::new(u64::MAX / 2);
        let mut pointers = Vec::new();

        for size in &sizes {
            if let Ok(pointer) = limiter.try_alloc(*size) {
                pointers.push(pointer);
            }
        }

        for pointer in pointers {
            limiter.free(pointer);
        }

        prop_assert_eq!(limiter.pod_memory_used(), 0);
        prop_assert_eq!(limiter.tracked_total(), 0);
        prop_assert_eq!(limiter.allocation_count(), 0);
        prop_assert_eq!(limiter.proc_usage(), 0, "proc_usage must be 0 after freeing all");
    }
}

// --- Pitched allocation proptest (hipMallocPitch / hipMalloc3D two-phase pattern) ---

#[derive(Debug, Clone)]
enum PitchedOperation {
    /// (estimated_size, overhead_pct): actual = estimated + estimated * overhead_pct / 100
    PitchedAlloc(u64, u8),
    /// Plain alloc (to interleave with pitched)
    PlainAlloc(u64),
    /// Free by index into live_pointers
    Free(usize),
}

fn pitched_operation_strategy() -> impl Strategy<Value = PitchedOperation> {
    prop_oneof![
        // Pitched: estimated 1-500KB, overhead 0-50% (models pitch alignment)
        (1u64..500_000, 0u8..50).prop_map(|(est, pct)| PitchedOperation::PitchedAlloc(est, pct)),
        // Plain alloc
        (1u64..500_000).prop_map(PitchedOperation::PlainAlloc),
        // Free
        (0usize..500).prop_map(PitchedOperation::Free),
    ]
}

proptest! {
    /// Random mix of pitched and plain alloc/free operations.
    /// pod_memory_used must always equal the sum of live allocation sizes,
    /// and pitched allocations must be tracked at their actual size (not estimated).
    #[test]
    fn pitched_alloc_accounting(operations in proptest::collection::vec(pitched_operation_strategy(), 1..500)) {
        let limiter = SimulatedLimiter::new(10_000_000);
        let mut live_pointers: Vec<usize> = Vec::new();

        for operation in &operations {
            match operation {
                PitchedOperation::PitchedAlloc(estimated, overhead_pct) => {
                    let extra = *estimated * (*overhead_pct as u64) / 100;
                    let actual = *estimated + extra;
                    if let Ok(pointer) = limiter.try_alloc_pitched(*estimated, actual, true) {
                        live_pointers.push(pointer);
                    }
                }
                PitchedOperation::PlainAlloc(size) => {
                    if let Ok(pointer) = limiter.try_alloc(*size) {
                        if *size > 0 {
                            live_pointers.push(pointer);
                        }
                    }
                }
                PitchedOperation::Free(index) => {
                    if !live_pointers.is_empty() {
                        let idx = *index % live_pointers.len();
                        let pointer = live_pointers.swap_remove(idx);
                        limiter.free(pointer);
                    }
                }
            }
        }

        prop_assert_eq!(
            limiter.allocation_count(),
            live_pointers.len(),
            "allocation_count must match live pointers"
        );
        prop_assert_eq!(
            limiter.pod_memory_used(),
            limiter.tracked_total(),
            "pod_memory_used must equal tracked_total"
        );
        prop_assert_eq!(limiter.proc_usage(), limiter.pod_memory_used(),
            "proc_usage must equal pod_memory_used in single-threaded scenario");
    }

    /// Pitched allocations must never push pod_memory_used above the limit,
    /// even with interleaved frees returning capacity to the pool.
    #[test]
    fn pitched_never_exceeds_limit(
        operations in proptest::collection::vec(pitched_operation_strategy(), 1..200)
    ) {
        let limit = 2_000_000u64;
        let limiter = SimulatedLimiter::new(limit);
        let mut live_pointers: Vec<usize> = Vec::new();

        for operation in &operations {
            match operation {
                PitchedOperation::PitchedAlloc(estimated, overhead_pct) => {
                    let extra = *estimated * (*overhead_pct as u64) / 100;
                    let actual = *estimated + extra;
                    if let Ok(pointer) = limiter.try_alloc_pitched(*estimated, actual, true) {
                        live_pointers.push(pointer);
                    }
                }
                PitchedOperation::PlainAlloc(size) => {
                    if let Ok(pointer) = limiter.try_alloc(*size) {
                        if *size > 0 {
                            live_pointers.push(pointer);
                        }
                    }
                }
                PitchedOperation::Free(index) => {
                    if !live_pointers.is_empty() {
                        let idx = *index % live_pointers.len();
                        let pointer = live_pointers.swap_remove(idx);
                        limiter.free(pointer);
                    }
                }
            }
            prop_assert!(
                limiter.pod_memory_used() <= limit,
                "pod_memory_used ({}) exceeded limit ({})",
                limiter.pod_memory_used(),
                limit
            );
        }
    }

    /// Pitched alloc then free: all memory must be reclaimed at actual_size,
    /// not estimated_size. Exercises that the tracker stores the right value.
    #[test]
    fn pitched_free_returns_to_zero(
        entries in proptest::collection::vec((1u64..500_000, 0u8..50), 1..50)
    ) {
        let limiter = SimulatedLimiter::new(u64::MAX / 2);
        let mut pointers = Vec::new();

        for (estimated, overhead_pct) in &entries {
            let extra = *estimated * (*overhead_pct as u64) / 100;
            let actual = *estimated + extra;
            if let Ok(pointer) = limiter.try_alloc_pitched(*estimated, actual, true) {
                pointers.push(pointer);
            }
        }

        for pointer in pointers {
            limiter.free(pointer);
        }

        prop_assert_eq!(limiter.pod_memory_used(), 0);
        prop_assert_eq!(limiter.tracked_total(), 0);
        prop_assert_eq!(limiter.allocation_count(), 0);
        prop_assert_eq!(limiter.proc_usage(), 0, "proc_usage must be 0 after freeing all pitched allocs");
    }
}

// --- Multi-device proptest ---

#[derive(Debug, Clone)]
enum MultiDeviceOp {
    Alloc { device_idx: usize, size: u64 },
    Free(usize),
}

fn multi_device_op_strategy(num_devices: usize) -> impl Strategy<Value = MultiDeviceOp> {
    prop_oneof![
        (0..num_devices, 1u64..1_000_000)
            .prop_map(|(device_idx, size)| MultiDeviceOp::Alloc { device_idx, size }),
        (0usize..500).prop_map(MultiDeviceOp::Free),
    ]
}

proptest! {
    /// Multi-device: random alloc/free across 3 devices. Each device's
    /// pod_memory_used must independently equal its own live allocation sum,
    /// and freeing a pointer must decrement the correct device's counter.
    #[test]
    fn multi_device_independent_accounting(
        operations in proptest::collection::vec(multi_device_op_strategy(3), 1..500)
    ) {
        let limiter = MultiDeviceSimulatedLimiter::new(&[5_000_000, 5_000_000, 5_000_000]);
        // (external_pointer, device_idx, size)
        let mut live: Vec<(usize, usize, u64)> = Vec::new();

        for op in &operations {
            match op {
                MultiDeviceOp::Alloc { device_idx, size } => {
                    if let Ok(ptr) = limiter.try_alloc(*device_idx, *size) {
                        live.push((ptr, *device_idx, *size));
                    }
                }
                MultiDeviceOp::Free(index) => {
                    if !live.is_empty() {
                        let idx = *index % live.len();
                        let (ptr, _, _) = live.swap_remove(idx);
                        limiter.free(ptr);
                    }
                }
            }
        }

        // Per-device invariant: pod_memory_used == sum of live alloc sizes for that device
        for device_idx in 0..3 {
            let expected: u64 = live.iter()
                .filter(|(_, d, _)| *d == device_idx)
                .map(|(_, _, size)| *size)
                .sum();
            prop_assert_eq!(
                limiter.pod_memory_used(device_idx),
                expected,
                "device {} pod_memory_used mismatch", device_idx
            );
            prop_assert_eq!(
                limiter.proc_usage(device_idx),
                expected,
                "device {} proc_usage mismatch", device_idx
            );
        }
    }

    /// Multi-device: per-device limits are independent. Filling device 0
    /// must not affect device 1's capacity.
    #[test]
    fn multi_device_limits_independent(
        sizes_d0 in proptest::collection::vec(1u64..500_000, 1..50),
        sizes_d1 in proptest::collection::vec(1u64..500_000, 1..50),
    ) {
        let limiter = MultiDeviceSimulatedLimiter::new(&[2_000_000, 2_000_000]);

        for size in &sizes_d0 {
            let _ = limiter.try_alloc(0, *size);
            prop_assert!(
                limiter.pod_memory_used(0) <= 2_000_000,
                "device 0 exceeded its limit"
            );
        }

        for size in &sizes_d1 {
            let _ = limiter.try_alloc(1, *size);
            prop_assert!(
                limiter.pod_memory_used(1) <= 2_000_000,
                "device 1 exceeded its limit"
            );
        }
    }

    /// drain_allocations on a single device: after random alloc/free, drain must
    /// return pod_memory_used to 0 and empty the tracker. Models the atexit handler.
    #[test]
    fn drain_returns_to_zero(operations in proptest::collection::vec(operation_strategy(), 1..500)) {
        let limiter = SimulatedLimiter::new(10_000_000);
        let mut live_pointers: Vec<usize> = Vec::new();

        for operation in &operations {
            match operation {
                Operation::Alloc(size) => {
                    if let Ok(pointer) = limiter.try_alloc(*size) {
                        if *size > 0 {
                            live_pointers.push(pointer);
                        }
                    }
                }
                Operation::Free(index) => {
                    if !live_pointers.is_empty() {
                        let idx = *index % live_pointers.len();
                        let pointer = live_pointers.swap_remove(idx);
                        limiter.free(pointer);
                    }
                }
            }
        }

        // drain_allocations must return sum of remaining live allocation sizes
        let used_before_drain = limiter.pod_memory_used();
        let drained = limiter.drain_allocations();
        prop_assert_eq!(drained, used_before_drain, "drained bytes must equal pod_memory_used before drain");
        prop_assert_eq!(limiter.pod_memory_used(), 0, "pod_memory_used must be 0 after drain");
        prop_assert_eq!(limiter.tracked_total(), 0, "tracked_total must be 0 after drain");
        prop_assert_eq!(limiter.allocation_count(), 0, "allocation_count must be 0 after drain");
        prop_assert_eq!(limiter.proc_usage(), 0, "proc_usage must be 0 after drain");
    }

    /// drain_allocations after partial free: alloc N, free some, drain the rest.
    /// Verifies drain only removes what's still tracked, not what was already freed.
    #[test]
    fn drain_after_partial_free(
        sizes in proptest::collection::vec(1u64..500_000, 2..50),
        free_fraction in 0u8..100,
    ) {
        let limiter = SimulatedLimiter::new(u64::MAX / 2);
        let mut pointers = Vec::new();

        for size in &sizes {
            if let Ok(pointer) = limiter.try_alloc(*size) {
                pointers.push(pointer);
            }
        }

        // Free a fraction of the allocations
        let num_to_free = (pointers.len() as u64 * free_fraction as u64 / 100) as usize;
        for pointer in pointers.drain(..num_to_free) {
            limiter.free(pointer);
        }

        let used_before_drain = limiter.pod_memory_used();
        let drained = limiter.drain_allocations();
        prop_assert_eq!(drained, used_before_drain);
        prop_assert_eq!(limiter.pod_memory_used(), 0);
        prop_assert_eq!(limiter.allocation_count(), 0);
        prop_assert_eq!(limiter.proc_usage(), 0, "proc_usage must be 0 after drain");
    }

    /// Multi-device drain: random alloc/free across devices, then drain all.
    /// Each device's counter must independently return to 0.
    #[test]
    fn multi_device_drain_returns_to_zero(
        operations in proptest::collection::vec(multi_device_op_strategy(3), 1..500)
    ) {
        let limiter = MultiDeviceSimulatedLimiter::new(&[5_000_000, 5_000_000, 5_000_000]);
        let mut live: Vec<usize> = Vec::new();

        for op in &operations {
            match op {
                MultiDeviceOp::Alloc { device_idx, size } => {
                    if let Ok(ptr) = limiter.try_alloc(*device_idx, *size) {
                        live.push(ptr);
                    }
                }
                MultiDeviceOp::Free(index) => {
                    if !live.is_empty() {
                        let idx = *index % live.len();
                        let ptr = live.swap_remove(idx);
                        limiter.free(ptr);
                    }
                }
            }
        }

        let drained = limiter.drain_allocations();
        // All devices must be at 0
        for device_idx in 0..3 {
            prop_assert_eq!(
                limiter.pod_memory_used(device_idx), 0,
                "device {} pod_memory_used must be 0 after drain", device_idx
            );
            prop_assert_eq!(
                limiter.proc_usage(device_idx), 0,
                "device {} proc_usage must be 0 after drain", device_idx
            );
        }
        // Drain total must be positive if there were any live allocations
        if !live.is_empty() {
            prop_assert!(drained > 0, "drained should be > 0 when there were live allocations");
        }
    }

    /// Multi-device: freeing a pointer on device 0 must not affect device 1's
    /// counter. Targeted test for cross-device free routing correctness.
    #[test]
    fn multi_device_free_does_not_affect_other_device(
        sizes_d0 in proptest::collection::vec(1u64..500_000, 1..20),
        sizes_d1 in proptest::collection::vec(1u64..500_000, 1..20),
    ) {
        let limiter = MultiDeviceSimulatedLimiter::new(&[10_000_000, 10_000_000]);
        let mut ptrs_d0 = Vec::new();
        let mut ptrs_d1 = Vec::new();

        // Allocate on both devices
        for size in &sizes_d0 {
            if let Ok(ptr) = limiter.try_alloc(0, *size) {
                ptrs_d0.push(ptr);
            }
        }
        for size in &sizes_d1 {
            if let Ok(ptr) = limiter.try_alloc(1, *size) {
                ptrs_d1.push(ptr);
            }
        }

        let d1_used_before = limiter.pod_memory_used(1);

        // Free all device 0 pointers — device 1 must be unchanged
        for ptr in &ptrs_d0 {
            limiter.free(*ptr);
        }

        prop_assert_eq!(limiter.pod_memory_used(0), 0, "device 0 should be empty after freeing all");
        prop_assert_eq!(limiter.proc_usage(0), 0, "device 0 proc_usage should be empty after freeing all");
        prop_assert_eq!(
            limiter.pod_memory_used(1),
            d1_used_before,
            "device 1 must be unchanged after freeing device 0 pointers"
        );
        prop_assert_eq!(
            limiter.proc_usage(1),
            d1_used_before,
            "device 1 proc_usage must be unchanged after freeing device 0 pointers"
        );
    }

    /// Reap-on-OOM on a specific device recovers stale usage and allows retry,
    /// without affecting other devices.
    #[test]
    fn multi_device_reap_on_oom_recovers_stale_device(
        real_size in 700_000u64..900_000,
        stale_size in 150_000u64..300_000,
        alloc_size in 10_000u64..100_000,
    ) {
        // Ensure real + stale > limit so the first try_alloc definitely fails
        // and the reap path is exercised.
        prop_assume!(real_size + stale_size + alloc_size > 1_000_000);

        let limiter = MultiDeviceSimulatedLimiter::new(&[1_000_000, 1_000_000]);

        // Device 0: real allocation
        let ptr0 = limiter.try_alloc(0, real_size).unwrap();

        // Device 0: inject stale usage (models dead process)
        limiter.inject_stale_usage(0, stale_size);

        // Device 1: unrelated allocation — should be untouched
        let ptr1 = limiter.try_alloc(1, 500_000).unwrap();
        let d1_before = limiter.pod_memory_used(1);

        // try_alloc_with_reap: first attempt fails (over limit),
        // reap recovers stale_size, retry may succeed
        let result = limiter.try_alloc_with_reap(0, alloc_size, |l| {
            l.recover_stale_usage(0, stale_size);
            1
        });

        if real_size + alloc_size <= 1_000_000 {
            // After reap removed stale, real + alloc fits — should succeed
            prop_assert!(result.is_ok(), "reap should recover enough for alloc");
            prop_assert_eq!(
                limiter.pod_memory_used(0),
                real_size + alloc_size,
                "device 0 should reflect real + new alloc (stale recovered)"
            );
        } else {
            // Even after reap, real + alloc > limit — still fails
            prop_assert!(result.is_err());
            prop_assert_eq!(
                limiter.pod_memory_used(0),
                real_size, // stale was recovered but alloc was denied
                "device 0 should reflect only real alloc (stale recovered, new denied)"
            );
        }

        // Device 1 must be untouched regardless
        prop_assert_eq!(
            limiter.pod_memory_used(1),
            d1_before,
            "device 1 must be unaffected by device 0 reap"
        );

        // Cleanup
        limiter.free(ptr0);
        limiter.free(ptr1);
        if let Ok(ptr) = result {
            limiter.free(ptr);
        }
    }
}

// --- Effective memory limit proptests ---

proptest! {
    /// When effective_mem_limit is set (non-zero, below mem_limit),
    /// pod_memory_used must never exceed the effective limit.
    #[test]
    fn effective_limit_gates_alloc(
        mem_limit in 1000u64..100_000,
        effective_ratio in 0.1f64..0.9,
        sizes in proptest::collection::vec(1u64..500, 1..50),
    ) {
        let limiter = SimulatedLimiter::new(mem_limit);
        let effective = (mem_limit as f64 * effective_ratio) as u64;
        limiter.set_effective_mem_limit(effective);

        let mut ptrs = Vec::new();
        for size in sizes {
            match limiter.try_alloc(size) {
                Ok(ptr) => ptrs.push(ptr),
                Err(()) => {}
            }
            // Invariant: pod_memory_used never exceeds effective limit
            prop_assert!(limiter.pod_memory_used() <= effective,
                "pod_memory_used {} exceeded effective limit {}",
                limiter.pod_memory_used(), effective);
        }
        // Cleanup
        for ptr in ptrs { limiter.free(ptr); }
        prop_assert_eq!(limiter.pod_memory_used(), 0);
    }

    /// When effective_mem_limit is 0 (default / not yet computed),
    /// the limiter falls back to mem_limit as the allocation ceiling.
    #[test]
    fn effective_limit_zero_falls_back_to_mem_limit(
        mem_limit in 1000u64..100_000,
        sizes in proptest::collection::vec(1u64..500, 1..50),
    ) {
        let limiter = SimulatedLimiter::new(mem_limit);
        // effective_mem_limit is 0 by default — should use mem_limit
        prop_assert_eq!(limiter.effective_mem_limit(), 0);

        let mut ptrs = Vec::new();
        for size in sizes {
            match limiter.try_alloc(size) {
                Ok(ptr) => ptrs.push(ptr),
                Err(()) => {}
            }
            prop_assert!(limiter.pod_memory_used() <= mem_limit);
        }
        for ptr in ptrs { limiter.free(ptr); }
        prop_assert_eq!(limiter.pod_memory_used(), 0);
    }
}
