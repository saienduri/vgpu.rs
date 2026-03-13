use hip_limiter_fuzz::{MultiDeviceSimulatedLimiter, SimulatedLimiter};

/// Zero-size allocation should succeed but not be tracked.
/// This matches the real limiter: `check_and_alloc!` only records when
/// `$request_size > 0`.
#[test]
fn zero_size_alloc_succeeds_but_not_tracked() {
    let limiter = SimulatedLimiter::new(1024);
    let pointer = limiter.try_alloc(0).expect("zero-size should succeed");
    assert!(pointer > 0, "should return a valid fake pointer");
    assert_eq!(limiter.pod_memory_used(), 0, "zero-size should not affect accounting");
    assert_eq!(limiter.allocation_count(), 0, "zero-size should not be tracked");

    // Freeing a zero-size allocation returns false (it was never tracked)
    assert!(!limiter.free(pointer), "zero-size alloc was not tracked, free returns false");
}

/// u64::MAX size must be denied due to saturating_add overflow check.
/// The check is `used.saturating_add(size) > mem_limit`. When size is u64::MAX,
/// saturating_add returns u64::MAX regardless of `used`, which exceeds any
/// practical mem_limit.
#[test]
fn max_u64_size_denied() {
    let limiter = SimulatedLimiter::new(1024 * 1024 * 1024);
    assert!(limiter.try_alloc(u64::MAX).is_err(), "u64::MAX should be denied");
    assert_eq!(limiter.pod_memory_used(), 0, "no accounting change on denial");
}

/// Freeing an unknown pointer returns false and does not change accounting.
#[test]
fn free_unknown_pointer_returns_false() {
    let limiter = SimulatedLimiter::new(1024);
    let pointer = limiter.try_alloc(100).expect("should succeed");
    assert_eq!(limiter.pod_memory_used(), 100);

    // Free a pointer that was never allocated
    assert!(!limiter.free(9999), "unknown pointer should return false");
    assert_eq!(limiter.pod_memory_used(), 100, "accounting unchanged");
    assert_eq!(limiter.allocation_count(), 1, "tracker unchanged");

    // Original allocation still freeable
    assert!(limiter.free(pointer));
    assert_eq!(limiter.pod_memory_used(), 0);
}

/// Double free: second free returns false, no accounting change.
#[test]
fn double_free_safe() {
    let limiter = SimulatedLimiter::new(1024);
    let pointer = limiter.try_alloc(256).expect("should succeed");
    assert_eq!(limiter.pod_memory_used(), 256);

    assert!(limiter.free(pointer), "first free should succeed");
    assert_eq!(limiter.pod_memory_used(), 0);

    assert!(!limiter.free(pointer), "second free should return false");
    assert_eq!(limiter.pod_memory_used(), 0, "no underflow from double free");
}

/// Allocating exactly at the limit boundary should succeed.
#[test]
fn alloc_at_exact_limit_boundary() {
    let limit = 1024u64;
    let limiter = SimulatedLimiter::new(limit);

    let pointer = limiter.try_alloc(limit).expect("exact limit should succeed");
    assert_eq!(limiter.pod_memory_used(), limit);

    // Next allocation of any size > 0 should be denied
    assert!(limiter.try_alloc(1).is_err(), "over limit by 1 byte should fail");
    assert_eq!(limiter.pod_memory_used(), limit, "denied alloc should not change accounting");

    limiter.free(pointer);
    assert_eq!(limiter.pod_memory_used(), 0);
}

/// Multiple small allocations that sum exactly to the limit.
#[test]
fn incremental_fill_to_limit() {
    let limit = 1000u64;
    let limiter = SimulatedLimiter::new(limit);
    let mut pointers = Vec::new();

    for _ in 0..10 {
        let pointer = limiter.try_alloc(100).expect("should fit");
        pointers.push(pointer);
    }
    assert_eq!(limiter.pod_memory_used(), 1000);

    // One more should be denied
    assert!(limiter.try_alloc(1).is_err());

    // Free one, then we can allocate again
    limiter.free(pointers.pop().expect("should have pointers"));
    assert_eq!(limiter.pod_memory_used(), 900);

    let pointer = limiter.try_alloc(100).expect("should fit after free");
    pointers.push(pointer);
    assert_eq!(limiter.pod_memory_used(), 1000);
}

/// Rapid alloc/free cycles should not leak memory in the accounting.
/// 10k iterations amplifies any per-cycle drift in pod_memory_used or
/// allocation_tracker — even a single leaked byte would accumulate to
/// a detectable non-zero final value.
#[test]
fn rapid_alloc_free_no_leak() {
    let limiter = SimulatedLimiter::new(1_000_000);

    for _ in 0..10_000 {
        let pointer = limiter.try_alloc(100).expect("should succeed");
        assert!(limiter.free(pointer));
    }

    assert_eq!(limiter.pod_memory_used(), 0);
    assert_eq!(limiter.allocation_count(), 0);
}

/// Rollback-on-native-failure: reservation succeeds but native allocator fails.
/// pod_memory_used must return to its previous value.
///
/// Gap: The check_and_alloc! macro's "native alloc fails after reservation" path
/// (mem.rs lines 44-48) was completely untested.
#[test]
fn native_failure_rolls_back_reservation() {
    let limiter = SimulatedLimiter::new(10_000);

    // Pre-fill with a tracked allocation
    let ptr = limiter.try_alloc(3_000).expect("should succeed");
    assert_eq!(limiter.pod_memory_used(), 3_000);

    // Native failure should leave pod_memory_used unchanged
    assert!(limiter.try_alloc_native_fails(2_000).is_err());
    assert_eq!(limiter.pod_memory_used(), 3_000, "rollback must restore previous value");
    assert_eq!(limiter.allocation_count(), 1, "no new allocation tracked");

    // The original allocation should still be freeable
    assert!(limiter.free(ptr));
    assert_eq!(limiter.pod_memory_used(), 0);
}

/// Multi-device: allocations on different devices have independent accounting.
/// Freeing a pointer decrements the correct device's counter.
///
/// Gap: Single-device SimulatedLimiter couldn't catch cross-device accounting bugs.
#[test]
fn multi_device_independent_accounting() {
    let limiter = MultiDeviceSimulatedLimiter::new(&[10_000, 20_000]);

    let ptr_d0 = limiter.try_alloc(0, 5_000).expect("device 0 alloc");
    let _ptr_d1 = limiter.try_alloc(1, 8_000).expect("device 1 alloc");

    assert_eq!(limiter.pod_memory_used(0), 5_000);
    assert_eq!(limiter.pod_memory_used(1), 8_000);

    // Free device 0's pointer — only device 0's counter decrements
    assert!(limiter.free(ptr_d0));
    assert_eq!(limiter.pod_memory_used(0), 0);
    assert_eq!(limiter.pod_memory_used(1), 8_000);

    // Device 0 limit is independent — can fill to its own limit
    let _ptr = limiter.try_alloc(0, 10_000).expect("device 0 at limit");
    assert!(limiter.try_alloc(0, 1).is_err(), "device 0 over limit");
    // Device 1 still has room
    let _ptr2 = limiter.try_alloc(1, 12_000).expect("device 1 still has room");
}

/// Allocations > MAX_ALLOC_SIZE (u64::MAX / 2) are rejected before fetch_add
/// to prevent transient wrapping of the atomic counter.
#[test]
fn max_alloc_size_guard_rejects_before_fetch_add() {
    let limiter = SimulatedLimiter::new(u64::MAX); // huge limit — guard should still reject
    let boundary = u64::MAX / 2;

    // Exactly at boundary should succeed (within limit)
    let ptr = limiter.try_alloc(boundary).expect("exactly MAX_ALLOC_SIZE should succeed");
    assert_eq!(limiter.pod_memory_used(), boundary);
    limiter.free(ptr);

    // One byte over boundary should be rejected by the guard
    assert!(limiter.try_alloc(boundary + 1).is_err(), "MAX_ALLOC_SIZE + 1 should be denied");
    assert_eq!(limiter.pod_memory_used(), 0, "guard rejects before fetch_add, no accounting change");
}

// --- Pitched allocation tests (hipMallocPitch / hipMalloc3D two-phase pattern) ---

/// Pitched allocation where pitch == width (no alignment overhead).
/// Accounting should match a simple try_alloc of the same size.
#[test]
fn pitched_alloc_no_overhead() {
    let limiter = SimulatedLimiter::new(10_000);
    // width=100, height=50 => estimated=5000, pitch==width => actual=5000
    let ptr = limiter.try_alloc_pitched(5_000, 5_000, true).expect("should succeed");
    assert_eq!(limiter.pod_memory_used(), 5_000);
    assert_eq!(limiter.allocation_count(), 1);

    assert!(limiter.free(ptr));
    assert_eq!(limiter.pod_memory_used(), 0);
}

/// Pitched allocation where pitch > width (alignment overhead fits within limit).
/// pod_memory_used must reflect actual_size (pitch * height), not estimated (width * height).
#[test]
fn pitched_alloc_with_overhead_within_limit() {
    let limiter = SimulatedLimiter::new(10_000);
    // width=100, height=50 => estimated=5000; pitch=128 => actual=6400
    let ptr = limiter.try_alloc_pitched(5_000, 6_400, true).expect("should succeed");
    assert_eq!(limiter.pod_memory_used(), 6_400, "must account for actual pitch*height");
    assert_eq!(limiter.allocation_count(), 1);

    assert!(limiter.free(ptr));
    assert_eq!(limiter.pod_memory_used(), 0);
}

/// Pitched allocation where alignment overhead pushes over the limit.
/// The estimated size fits, but actual_size (with pitch alignment) does not.
/// Both the initial reservation and the extra must be rolled back.
#[test]
fn pitched_alloc_overhead_exceeds_limit() {
    let limiter = SimulatedLimiter::new(6_000);
    // estimated=5000 fits within 6000, but actual=6400 exceeds limit
    assert!(limiter.try_alloc_pitched(5_000, 6_400, true).is_err());
    assert_eq!(limiter.pod_memory_used(), 0, "rollback must restore to zero");
    assert_eq!(limiter.allocation_count(), 0);
}

/// Pitched allocation where alignment overhead pushes over limit with pre-existing allocations.
/// Verifies rollback restores pod_memory_used to its pre-pitched-alloc value (not zero).
#[test]
fn pitched_alloc_overhead_rollback_preserves_existing() {
    let limiter = SimulatedLimiter::new(8_000);
    let ptr1 = limiter.try_alloc(3_000).expect("pre-fill");
    assert_eq!(limiter.pod_memory_used(), 3_000);

    // estimated=4000 fits (3000+4000=7000 <= 8000)
    // actual=5500 doesn't (3000+5500=8500 > 8000)
    assert!(limiter.try_alloc_pitched(4_000, 5_500, true).is_err());
    assert_eq!(limiter.pod_memory_used(), 3_000, "rollback must restore to pre-alloc value");
    assert_eq!(limiter.allocation_count(), 1, "only the pre-fill allocation remains");

    assert!(limiter.free(ptr1));
    assert_eq!(limiter.pod_memory_used(), 0);
}

/// Pitched allocation where native allocator fails after reservation.
/// Reservation must be rolled back regardless of overhead.
#[test]
fn pitched_alloc_native_failure_rolls_back() {
    let limiter = SimulatedLimiter::new(10_000);
    // Native fails — estimated reservation must be undone
    assert!(limiter.try_alloc_pitched(5_000, 6_400, false).is_err());
    assert_eq!(limiter.pod_memory_used(), 0, "native failure must rollback reservation");
    assert_eq!(limiter.allocation_count(), 0);
}

/// Zero-size pitched allocation succeeds but is not tracked (same as regular zero-size).
#[test]
fn pitched_alloc_zero_size() {
    let limiter = SimulatedLimiter::new(1_000);
    let ptr = limiter.try_alloc_pitched(0, 0, true).expect("zero-size should succeed");
    assert!(ptr > 0);
    assert_eq!(limiter.pod_memory_used(), 0);
    assert_eq!(limiter.allocation_count(), 0);
}

/// Pitched allocation for 3D: verifies the same two-phase pattern works with 3D sizes.
/// estimated = width * height * depth, actual = pitch * height * depth
#[test]
fn pitched_alloc_3d_overhead_within_limit() {
    let limiter = SimulatedLimiter::new(100_000);
    // width=100, height=50, depth=10 => estimated=50_000
    // pitch=128 => actual=128*50*10=64_000
    let ptr = limiter.try_alloc_pitched(50_000, 64_000, true).expect("should succeed");
    assert_eq!(limiter.pod_memory_used(), 64_000);

    assert!(limiter.free(ptr));
    assert_eq!(limiter.pod_memory_used(), 0);
}

/// Pitched allocation for 3D: overhead exceeds limit.
/// depth multiplier amplifies the alignment overhead.
#[test]
fn pitched_alloc_3d_overhead_exceeds_limit() {
    let limiter = SimulatedLimiter::new(55_000);
    // estimated=50_000 fits, but actual=64_000 exceeds 55_000
    assert!(limiter.try_alloc_pitched(50_000, 64_000, true).is_err());
    assert_eq!(limiter.pod_memory_used(), 0, "3D rollback must restore to zero");
}

/// Pitched allocation where actual_size exceeds MAX_ALLOC_SIZE (u64::MAX / 2).
/// The estimated_size is within bounds but the actual alignment overhead pushes
/// actual_size over the MAX_ALLOC_SIZE guard.
///
/// In the simulated limiter, the MAX_ALLOC_SIZE guard on actual_size fires
/// before Phase 1 (reservation). In the real detour code, actual_size is only
/// known after the native call — the equivalent protection comes from
/// checked_mul returning None (overflow), which triggers rollback + free.
/// Both paths produce the same outcome: no net reservation, allocation denied.
#[test]
fn pitched_alloc_actual_size_exceeds_max_alloc() {
    let limiter = SimulatedLimiter::new(u64::MAX);
    let max_alloc = u64::MAX / 2;
    // estimated fits, but actual exceeds MAX_ALLOC_SIZE
    assert!(limiter.try_alloc_pitched(max_alloc - 1, max_alloc + 1, true).is_err());
    assert_eq!(limiter.pod_memory_used(), 0, "denial must leave accounting unchanged");
    assert_eq!(limiter.allocation_count(), 0);
}

/// When used + size overflows u64 (used is large, size is large but realistic),
/// saturating_add caps at u64::MAX and the alloc is denied without corrupting
/// pod_memory_used. Distinct from max_u64_size_denied which tests from empty.
#[test]
fn saturating_add_prevents_overflow_when_near_full() {
    let limit = 1_000_000u64;
    let limiter = SimulatedLimiter::new(limit);

    // Fill most of the limit
    let pointer = limiter.try_alloc(999_999).expect("should fit");
    assert_eq!(limiter.pod_memory_used(), 999_999);

    // Request more than remaining — denied, accounting unchanged
    assert!(limiter.try_alloc(2).is_err());
    assert_eq!(limiter.pod_memory_used(), 999_999, "no change on denied alloc");

    // Exactly 1 byte remaining — should succeed
    let pointer2 = limiter.try_alloc(1).expect("exactly 1 byte left");
    assert_eq!(limiter.pod_memory_used(), 1_000_000);

    limiter.free(pointer);
    limiter.free(pointer2);
    assert_eq!(limiter.pod_memory_used(), 0);
}
