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
                "device {} must be 0 after drain", device_idx
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
        prop_assert_eq!(
            limiter.pod_memory_used(1),
            d1_used_before,
            "device 1 must be unchanged after freeing device 0 pointers"
        );
    }
}
