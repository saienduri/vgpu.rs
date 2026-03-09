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
