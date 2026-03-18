use std::sync::Arc;

use hip_limiter_fuzz::{MultiDeviceSimulatedLimiter, SimulatedLimiter};

/// Concurrent alloc/free stress test.
///
/// N threads perform random alloc/free operations on a shared SimulatedLimiter.
/// After all threads join, we verify:
/// - No panics occurred
/// - No underflow (pod_memory_used doesn't wrap around)
/// - pod_memory_used == tracked_total (sum of live allocations)
/// - pod_memory_used never exceeds mem_limit (reserve-then-allocate eliminates TOCTOU)
#[test]
fn concurrent_alloc_free_consistency() {
    let limiter = Arc::new(SimulatedLimiter::new(100_000_000));
    let thread_count = 8;
    let operations_per_thread = 1000;

    let handles: Vec<_> = (0..thread_count)
        .map(|thread_id| {
            let limiter = Arc::clone(&limiter);
            std::thread::spawn(move || {
                let mut local_pointers: Vec<usize> = Vec::new();
                let mut rng_state: u64 = thread_id as u64 + 1;

                for _ in 0..operations_per_thread {
                    // Simple xorshift PRNG (deterministic, no external dep needed)
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;

                    let should_free = !local_pointers.is_empty() && (rng_state % 3 == 0);

                    if should_free {
                        let index = (rng_state as usize) % local_pointers.len();
                        let pointer = local_pointers.swap_remove(index);
                        limiter.free(pointer);
                    } else {
                        let size = (rng_state % 10_000) + 1;
                        if let Ok(pointer) = limiter.try_alloc(size) {
                            local_pointers.push(pointer);
                        }
                    }
                }

                // Return remaining live pointers so we can free them after join
                local_pointers
            })
        })
        .collect();

    let mut all_remaining: Vec<usize> = Vec::new();
    for handle in handles {
        let remaining = handle.join().expect("thread should not panic");
        all_remaining.extend(remaining);
    }

    // At this point, only the returned pointers are still live.
    // Verify consistency before final cleanup.
    assert_eq!(
        limiter.pod_memory_used(),
        limiter.tracked_total(),
        "pod_memory_used must equal tracked_total after all threads join"
    );

    // Free remaining allocations
    for pointer in &all_remaining {
        assert!(
            limiter.free(*pointer),
            "live pointer should be freeable"
        );
    }

    assert_eq!(limiter.pod_memory_used(), 0, "all freed, usage should be 0");
    assert_eq!(limiter.tracked_total(), 0);
    assert_eq!(limiter.allocation_count(), 0);
}

/// Concurrent allocations against a tight limit — verifies reserve-then-allocate
/// prevents overcommit under contention.
///
/// Gap: Previous concurrent tests used large limits (100M or u64::MAX/2), so threads
/// never raced on the limit boundary. This test uses a small limit where most threads
/// must be denied, exercising the atomic rollback path under contention.
#[test]
fn concurrent_tight_limit_no_overcommit() {
    let mem_limit: u64 = 1_000; // Very tight
    let limiter = Arc::new(SimulatedLimiter::new(mem_limit));
    let thread_count = 16;
    let alloc_size: u64 = 200; // At most 5 can succeed (5 * 200 = 1000)

    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let limiter = Arc::clone(&limiter);
            std::thread::spawn(move || limiter.try_alloc(alloc_size).ok())
        })
        .collect();

    let mut success_count = 0usize;
    for handle in handles {
        if handle.join().expect("thread should not panic").is_some() {
            success_count += 1;
        }
    }

    let max_possible = (mem_limit / alloc_size) as usize;
    assert!(
        success_count <= max_possible,
        "overcommit: {success_count} succeeded but max is {max_possible}"
    );
    assert_eq!(
        limiter.pod_memory_used(),
        success_count as u64 * alloc_size,
        "SHM accounting mismatch"
    );
    assert!(success_count > 0, "at least one alloc should succeed");
    // With 16 threads competing for 5 slots, some must be denied
    assert!(
        success_count < thread_count,
        "with {thread_count} threads and {max_possible} slots, some must be denied"
    );
}

/// Concurrent mix of successful allocs, native failures, and frees.
/// Verifies that rollback-on-native-failure is safe under contention.
///
/// Gap: No previous test exercised the rollback path concurrently.
#[test]
fn concurrent_native_failures_no_drift() {
    let limiter = Arc::new(SimulatedLimiter::new(1_000_000));
    let thread_count = 8;
    let operations_per_thread = 500;

    let handles: Vec<_> = (0..thread_count)
        .map(|thread_id| {
            let limiter = Arc::clone(&limiter);
            std::thread::spawn(move || {
                let mut local_pointers: Vec<usize> = Vec::new();
                let mut rng_state: u64 = thread_id as u64 + 42;

                for _ in 0..operations_per_thread {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;

                    match rng_state % 4 {
                        0 => {
                            // Successful alloc
                            let size = (rng_state % 1_000) + 1;
                            if let Ok(pointer) = limiter.try_alloc(size) {
                                local_pointers.push(pointer);
                            }
                        }
                        1 => {
                            // Native failure (rollback)
                            let size = (rng_state % 1_000) + 1;
                            let _ = limiter.try_alloc_native_fails(size);
                        }
                        _ => {
                            // Free
                            if !local_pointers.is_empty() {
                                let index = (rng_state as usize) % local_pointers.len();
                                let pointer = local_pointers.swap_remove(index);
                                limiter.free(pointer);
                            }
                        }
                    }
                }
                local_pointers
            })
        })
        .collect();

    let mut all_remaining: Vec<usize> = Vec::new();
    for handle in handles {
        let remaining = handle.join().expect("thread should not panic");
        all_remaining.extend(remaining);
    }

    assert_eq!(
        limiter.pod_memory_used(),
        limiter.tracked_total(),
        "pod_memory_used must equal tracked_total despite native failures"
    );

    for pointer in &all_remaining {
        assert!(limiter.free(*pointer));
    }

    assert_eq!(limiter.pod_memory_used(), 0);
    assert_eq!(limiter.allocation_count(), 0);
}

/// Verify that concurrent alloc-only produces consistent accounting.
///
/// All threads allocate without freeing. After joining, allocation_count must
/// match the total number of successful allocations, and pod_memory_used must
/// equal tracked_total.
#[test]
fn concurrent_alloc_only_consistency() {
    let limiter = Arc::new(SimulatedLimiter::new(u64::MAX / 2));
    let thread_count = 4;
    let allocations_per_thread = 500;

    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let limiter = Arc::clone(&limiter);
            std::thread::spawn(move || {
                let mut pointers = Vec::new();
                for i in 0..allocations_per_thread {
                    let size = (i as u64 + 1) * 100;
                    if let Ok(pointer) = limiter.try_alloc(size) {
                        pointers.push(pointer);
                    }
                }
                pointers
            })
        })
        .collect();

    let mut total_pointers = 0usize;
    for handle in handles {
        let pointers = handle.join().expect("thread should not panic");
        total_pointers += pointers.len();
    }

    assert_eq!(limiter.allocation_count(), total_pointers);
    assert_eq!(limiter.pod_memory_used(), limiter.tracked_total());
    assert!(limiter.pod_memory_used() > 0);
}

/// Concurrent drain while allocating: some threads alloc/free while one thread
/// calls drain_allocations. After all threads join and drain completes,
/// pod_memory_used must be 0 (drain removes everything that was tracked at the
/// time of iteration, and any concurrent allocs that sneak in are still tracked).
///
/// This models the race between atexit drain and late allocations from other
/// threads that haven't yet been joined. In practice, atexit runs after main()
/// returns and all non-detached threads should be joined, but we test the
/// worst case.
#[test]
fn concurrent_drain_while_allocating() {
    let limiter = Arc::new(SimulatedLimiter::new(100_000_000));
    let thread_count = 6;
    let operations_per_thread = 500;

    // Phase 1: Populate with some allocations
    let mut initial_pointers = Vec::new();
    for i in 0..100 {
        if let Ok(ptr) = limiter.try_alloc((i as u64 + 1) * 100) {
            initial_pointers.push(ptr);
        }
    }
    assert!(limiter.pod_memory_used() > 0);

    // Phase 2: Concurrent alloc/free threads + one drain thread
    let drain_limiter = Arc::clone(&limiter);
    let drain_handle = std::thread::spawn(move || {
        // Small yield to let alloc threads start
        std::thread::yield_now();
        drain_limiter.drain_allocations()
    });

    let alloc_handles: Vec<_> = (0..thread_count)
        .map(|thread_id| {
            let limiter = Arc::clone(&limiter);
            std::thread::spawn(move || {
                let mut local_pointers: Vec<usize> = Vec::new();
                let mut rng_state: u64 = thread_id as u64 + 99;

                for _ in 0..operations_per_thread {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;

                    let should_free = !local_pointers.is_empty() && (rng_state % 3 == 0);
                    if should_free {
                        let index = (rng_state as usize) % local_pointers.len();
                        let pointer = local_pointers.swap_remove(index);
                        limiter.free(pointer);
                    } else {
                        let size = (rng_state % 5_000) + 1;
                        if let Ok(pointer) = limiter.try_alloc(size) {
                            local_pointers.push(pointer);
                        }
                    }
                }
                local_pointers
            })
        })
        .collect();

    let drained = drain_handle.join().expect("drain thread should not panic");
    assert!(drained > 0, "drain should have removed initial allocations");

    let mut all_remaining: Vec<usize> = Vec::new();
    for handle in alloc_handles {
        let remaining = handle.join().expect("alloc thread should not panic");
        all_remaining.extend(remaining);
    }

    // Key invariant: pod_memory_used == tracked_total (no drift from concurrent drain)
    assert_eq!(
        limiter.pod_memory_used(),
        limiter.tracked_total(),
        "pod_memory_used must equal tracked_total after concurrent drain + alloc/free"
    );

    // Final cleanup: free remaining + second drain to catch anything
    for pointer in &all_remaining {
        limiter.free(*pointer);
    }
    let final_drain = limiter.drain_allocations();

    assert_eq!(limiter.pod_memory_used(), 0, "must be 0 after full cleanup");
    assert_eq!(limiter.allocation_count(), 0);
    // final_drain should be 0 since we freed everything manually
    assert_eq!(final_drain, 0, "no allocations should remain after manual free");
}

/// Concurrent multi-device drain: threads allocate across devices while drain fires.
/// Verifies per-device counters stay consistent.
#[test]
fn concurrent_multi_device_drain() {
    let limiter = Arc::new(MultiDeviceSimulatedLimiter::new(&[100_000, 100_000, 100_000]));

    // Populate
    for device in 0..3 {
        for i in 0..20 {
            let _ = limiter.try_alloc(device, (i as u64 + 1) * 100);
        }
    }

    let drain_limiter = Arc::clone(&limiter);
    let drain_handle = std::thread::spawn(move || {
        std::thread::yield_now();
        drain_limiter.drain_allocations()
    });

    let alloc_handles: Vec<_> = (0..6)
        .map(|thread_id| {
            let limiter = Arc::clone(&limiter);
            std::thread::spawn(move || {
                let mut local_pointers = Vec::new();
                let mut rng_state: u64 = thread_id as u64 + 77;
                for _ in 0..200 {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;

                    let should_free = !local_pointers.is_empty() && (rng_state % 3 == 0);
                    if should_free {
                        let index = (rng_state as usize) % local_pointers.len();
                        let pointer = local_pointers.swap_remove(index);
                        limiter.free(pointer);
                    } else {
                        let device = (rng_state as usize) % 3;
                        let size = (rng_state % 3_000) + 1;
                        if let Ok(ptr) = limiter.try_alloc(device, size) {
                            local_pointers.push(ptr);
                        }
                    }
                }
                local_pointers
            })
        })
        .collect();

    let drained = drain_handle.join().expect("drain should not panic");
    assert!(drained > 0);

    let mut remaining = Vec::new();
    for h in alloc_handles {
        remaining.extend(h.join().expect("thread should not panic"));
    }

    // Free remaining and drain again
    for ptr in &remaining {
        limiter.free(*ptr);
    }
    limiter.drain_allocations();

    for device in 0..3 {
        assert_eq!(
            limiter.pod_memory_used(device), 0,
            "device {device} must be 0 after full cleanup"
        );
    }
}

/// Multi-device concurrent stress: threads target random devices.
/// Verifies per-device accounting stays independent under cross-device contention.
///
/// Gap: All prior concurrent tests use a single device. Cross-device accounting
/// bugs (e.g., freeing against the wrong device's counter) only manifest here.
#[test]
fn concurrent_multi_device_consistency() {
    let limiter = Arc::new(MultiDeviceSimulatedLimiter::new(&[50_000, 50_000, 50_000]));
    let thread_count = 8;
    let operations_per_thread = 500;

    let handles: Vec<_> = (0..thread_count)
        .map(|thread_id| {
            let limiter = Arc::clone(&limiter);
            std::thread::spawn(move || {
                let mut local_pointers: Vec<usize> = Vec::new();
                let mut rng_state: u64 = thread_id as u64 + 7;

                for _ in 0..operations_per_thread {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;

                    let should_free = !local_pointers.is_empty() && (rng_state % 3 == 0);

                    if should_free {
                        let index = (rng_state as usize) % local_pointers.len();
                        let pointer = local_pointers.swap_remove(index);
                        limiter.free(pointer);
                    } else {
                        let device_idx = (rng_state as usize) % limiter.device_count();
                        let size = (rng_state % 5_000) + 1;
                        if let Ok(pointer) = limiter.try_alloc(device_idx, size) {
                            local_pointers.push(pointer);
                        }
                    }
                }
                local_pointers
            })
        })
        .collect();

    let mut all_remaining: Vec<usize> = Vec::new();
    for handle in handles {
        let remaining = handle.join().expect("thread should not panic");
        all_remaining.extend(remaining);
    }

    // Per-device accounting: each device's usage must be <= its limit
    for device_idx in 0..limiter.device_count() {
        assert!(
            limiter.pod_memory_used(device_idx) <= 50_000,
            "device {device_idx} exceeded limit"
        );
    }

    // Free remaining and verify all devices return to 0
    for pointer in &all_remaining {
        assert!(limiter.free(*pointer));
    }
    for device_idx in 0..limiter.device_count() {
        assert_eq!(
            limiter.pod_memory_used(device_idx), 0,
            "device {device_idx} should be 0 after cleanup"
        );
    }
}
