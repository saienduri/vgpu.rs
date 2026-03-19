use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use dashmap::DashMap;

/// Maximum single allocation size — rejects before fetch_add to prevent
/// transient wrapping of the atomic counter.
const MAX_ALLOC_SIZE: u64 = u64::MAX / 2;

/// Atomically subtract `size` from `counter`, clamping at zero.
///
/// Uses a CAS loop to prevent underflow wrapping. Mirrors
/// `saturating_fetch_sub_pod_memory_used` on `SharedDeviceInfoV2`.
fn saturating_fetch_sub(counter: &AtomicU64, size: u64) {
    loop {
        let current = counter.load(Ordering::Acquire);
        let new_value = current.saturating_sub(size);
        if counter
            .compare_exchange_weak(current, new_value, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            break;
        }
    }
}

/// Pure-Rust model of the hip-limiter's memory accounting logic.
///
/// This faithfully reproduces the semantics of `limiter.rs` and the `check_and_alloc!`
/// macro from `detour/mem.rs`, without FFI, Frida, or GPU dependencies.
///
/// The real limiter uses a reserve-then-allocate pattern: atomically increment
/// `pod_memory_used` first (reserving the space), then check if the new total
/// exceeds the limit. If over limit, roll back the increment. This eliminates
/// the TOCTOU race that existed in the old check-then-allocate pattern.
///
/// NOTE: When `mem_limit == 0`, the real limiter reads mem_limit from SHM (which
/// could be any value set by the hypervisor). This model treats 0 as a literal
/// limit, so all non-zero allocations will be denied. This divergence is acceptable
/// because `mem_limit == 0` is not a valid production configuration.
pub struct SimulatedLimiter {
    mem_limit: u64,
    /// Mirrors the SHM `pod_memory_used` atomic counter.
    /// Updated via `fetch_add` on alloc and `fetch_sub` on free.
    pod_memory_used: AtomicU64,
    /// Mirrors the ProcSlot `used[device_idx]` counter for per-PID tracking.
    /// Single-device model, so one AtomicU64 suffices.
    proc_usage: AtomicU64,
    /// Maps fake pointer -> allocation size.
    /// Mirrors `allocation_tracker: DashMap<usize, (usize, u64)>` in the real limiter,
    /// but we omit the device_idx since we model a single device.
    allocation_tracker: DashMap<usize, u64>,
    /// Monotonically increasing counter to generate unique fake pointers.
    next_pointer: AtomicUsize,
}

impl SimulatedLimiter {
    pub fn new(mem_limit: u64) -> Self {
        Self {
            mem_limit,
            pod_memory_used: AtomicU64::new(0),
            proc_usage: AtomicU64::new(0),
            allocation_tracker: DashMap::new(),
            // Start at 1 so pointer 0 is never returned (mirrors real GPU behavior
            // where NULL/0 is reserved).
            next_pointer: AtomicUsize::new(1),
        }
    }

    /// Attempt an allocation of `size` bytes.
    ///
    /// Models the reserve-then-allocate flow in `check_and_alloc!`:
    /// 1. Atomically increment `pod_memory_used` by `size` (reserve)
    /// 2. If new total > `mem_limit`, roll back and deny (return Err)
    /// 3. Otherwise, "allocate" (generate fake ptr) and track in DashMap
    ///
    /// Zero-size allocations succeed but are not tracked, matching the real
    /// `check_and_alloc!` behavior: `if result == HIP_SUCCESS && $request_size > 0`.
    pub fn try_alloc(&self, size: u64) -> Result<usize, ()> {
        if size == 0 {
            // Returns a unique pointer but does not track it in allocation_tracker.
            // free(ptr) will return false. This matches the real limiter's behavior
            // where zero-size allocs skip reservation and tracking entirely.
            let pointer = self.next_pointer.fetch_add(1, Ordering::Relaxed);
            return Ok(pointer);
        }

        if size > MAX_ALLOC_SIZE {
            return Err(());
        }

        // Reserve first: atomically increment pod_memory_used
        let previous_used = self.pod_memory_used.fetch_add(size, Ordering::AcqRel);
        let new_used = previous_used.saturating_add(size);

        if new_used > self.mem_limit {
            // Over limit — roll back the reservation
            self.saturating_sub_pod_memory_used(size);
            return Err(());
        }

        let pointer = self.next_pointer.fetch_add(1, Ordering::Relaxed);
        self.allocation_tracker.insert(pointer, size);
        self.proc_usage.fetch_add(size, Ordering::AcqRel);
        Ok(pointer)
    }

    /// Free a previously allocated pointer.
    ///
    /// Returns `true` if the pointer was tracked (and accounting was decremented),
    /// `false` if the pointer was unknown (no accounting change).
    ///
    /// Models the free hook flow: native free succeeds (always, in this model),
    /// then `record_free` removes from DashMap and `fetch_sub` on `pod_memory_used`.
    /// In the real limiter, accounting is only updated after native free succeeds.
    pub fn free(&self, pointer: usize) -> bool {
        let Some((_, size)) = self.allocation_tracker.remove(&pointer) else {
            return false;
        };
        self.saturating_sub_pod_memory_used(size);
        self.saturating_sub_proc_usage(size);
        true
    }

    /// Returns the current value of the pod_memory_used counter.
    pub fn pod_memory_used(&self) -> u64 {
        self.pod_memory_used.load(Ordering::Acquire)
    }

    /// Returns the sum of all live allocation sizes in the tracker.
    /// In a race-free scenario, this equals `pod_memory_used()`.
    pub fn tracked_total(&self) -> u64 {
        self.allocation_tracker
            .iter()
            .map(|entry| *entry.value())
            .sum()
    }

    /// Returns the configured memory limit.
    pub fn mem_limit(&self) -> u64 {
        self.mem_limit
    }

    /// Returns the current per-PID usage (mirrors ProcSlot `used[device_idx]`).
    pub fn proc_usage(&self) -> u64 {
        self.proc_usage.load(Ordering::Acquire)
    }

    /// Returns the number of live tracked allocations.
    pub fn allocation_count(&self) -> usize {
        self.allocation_tracker.len()
    }

    /// Inject stale external usage into `pod_memory_used` without tracking it.
    ///
    /// Models a dead process that incremented the SHM counter but never freed.
    /// This usage is invisible to `tracked_total()` and `allocation_tracker`,
    /// just like real stale usage from a SIGKILL'd process.
    pub fn inject_stale_usage(&self, size: u64) {
        self.pod_memory_used.fetch_add(size, Ordering::AcqRel);
    }

    /// Recover stale usage from `pod_memory_used` (saturating subtract).
    ///
    /// Models the reap path: `reap_dead_pids` subtracts dead processes' usage
    /// from the SHM counter via `saturating_fetch_sub_pod_memory_used`.
    pub fn recover_stale_usage(&self, size: u64) {
        self.saturating_sub_pod_memory_used(size);
    }

    /// Attempt allocation with reap-on-OOM retry, mirroring the real `try_reserve` flow.
    ///
    /// 1. Try `try_alloc(size)`
    /// 2. If over limit, call `reap_fn` to recover stale capacity
    /// 3. If reap recovered anything (returned > 0), retry once
    /// 4. No second retry (prevents infinite recursion)
    pub fn try_alloc_with_reap(&self, size: u64, reap_fn: impl FnOnce(&Self) -> u64) -> Result<usize, ()> {
        match self.try_alloc(size) {
            Ok(ptr) => Ok(ptr),
            Err(()) => {
                let recovered = reap_fn(self);
                if recovered > 0 {
                    self.try_alloc(size)
                } else {
                    Err(())
                }
            }
        }
    }

    fn saturating_sub_pod_memory_used(&self, size: u64) {
        saturating_fetch_sub(&self.pod_memory_used, size);
    }

    fn saturating_sub_proc_usage(&self, size: u64) {
        saturating_fetch_sub(&self.proc_usage, size);
    }

    /// Simulate a pitched allocation (hipMallocPitch / hipMalloc3D).
    ///
    /// Models the two-phase reserve pattern:
    /// 1. Reserve `estimated_size` (width * height [* depth])
    /// 2. "Native allocator" returns `actual_size` (pitch * height [* depth], where pitch >= width)
    /// 3. If `actual_size > estimated_size`, try to reserve the extra overhead
    ///    - If that pushes over limit: rollback everything, return Err
    /// 4. Record allocation with `actual_size`
    ///
    /// `actual_size` must be >= `estimated_size` (pitch >= width invariant).
    /// If `native_succeeds` is false, simulates native allocator failure after reservation.
    pub fn try_alloc_pitched(
        &self,
        estimated_size: u64,
        actual_size: u64,
        native_succeeds: bool,
    ) -> Result<usize, ()> {
        debug_assert!(actual_size >= estimated_size, "pitch >= width invariant");

        if estimated_size == 0 {
            let pointer = self.next_pointer.fetch_add(1, Ordering::Relaxed);
            return Ok(pointer);
        }

        if estimated_size > MAX_ALLOC_SIZE || actual_size > MAX_ALLOC_SIZE {
            return Err(());
        }

        // Phase 1: Reserve estimated_size
        let previous_used = self.pod_memory_used.fetch_add(estimated_size, Ordering::AcqRel);
        let new_used = previous_used.saturating_add(estimated_size);

        if new_used > self.mem_limit {
            self.saturating_sub_pod_memory_used(estimated_size);
            return Err(());
        }

        // Phase 2: Native allocator
        if !native_succeeds {
            self.saturating_sub_pod_memory_used(estimated_size);
            return Err(());
        }

        // Phase 3: Reserve alignment overhead (extra = actual_size - estimated_size)
        let extra = actual_size.saturating_sub(estimated_size);

        if extra > 0 {
            let prev = self.pod_memory_used.fetch_add(extra, Ordering::AcqRel);
            let new_total = prev.saturating_add(extra);

            if new_total > self.mem_limit {
                // Extra overhead pushes over limit — rollback everything
                // (estimated_size + extra = actual_size, rolled back in one op)
                self.saturating_sub_pod_memory_used(estimated_size + extra);
                return Err(());
            }
        }

        // Phase 4: Record with actual_size
        let pointer = self.next_pointer.fetch_add(1, Ordering::Relaxed);
        self.allocation_tracker.insert(pointer, actual_size);
        self.proc_usage.fetch_add(actual_size, Ordering::AcqRel);
        Ok(pointer)
    }

    /// Drain all tracked allocations and decrement pod_memory_used.
    ///
    /// Models the real limiter's `drain_allocations()` atexit handler: iterates
    /// the allocation tracker, aggregates per-device totals, and does a bulk
    /// saturating_sub. After drain, pod_memory_used should be 0 and the tracker
    /// should be empty (for a single-device limiter, there's only one device).
    ///
    /// Returns the total bytes drained.
    pub fn drain_allocations(&self) -> u64 {
        let mut total: u64 = 0;
        self.allocation_tracker.retain(|_, size| {
            total += *size;
            false // remove every entry
        });

        if total > 0 {
            self.saturating_sub_pod_memory_used(total);
        }

        // Zero proc_usage (mirrors drain_our_slot zeroing the ProcSlot counters)
        self.proc_usage.store(0, Ordering::Release);

        total
    }

    /// Simulate an allocation where the native allocator fails after reservation.
    ///
    /// Models the check_and_alloc! path: try_reserve succeeds, but the native HIP call
    /// returns an error. The reservation must be rolled back so pod_memory_used is
    /// restored to its pre-reserve value.
    ///
    /// Gap: The rollback-on-native-failure path was previously untested in both the
    /// Rust fuzz suite and the Python CTS.
    pub fn try_alloc_native_fails(&self, size: u64) -> Result<(), ()> {
        if size == 0 {
            return Ok(());
        }

        if size > MAX_ALLOC_SIZE {
            return Err(());
        }

        // Reserve: atomically increment pod_memory_used
        let previous_used = self.pod_memory_used.fetch_add(size, Ordering::AcqRel);
        let new_used = previous_used.saturating_add(size);

        if new_used > self.mem_limit {
            // Over limit — roll back
            self.saturating_sub_pod_memory_used(size);
            return Err(());
        }

        // Simulate native failure — roll back the reservation
        self.saturating_sub_pod_memory_used(size);
        Err(())
    }
}

/// Multi-device wrapper around SimulatedLimiter.
///
/// Models the real limiter's per-device accounting: each device has its own
/// pod_memory_used counter and mem_limit. The allocation tracker maps
/// pointer -> (device_idx, size), matching the real DashMap<usize, (usize, u64)>.
///
/// Gap: The single-device SimulatedLimiter couldn't catch cross-device accounting
/// bugs (e.g., freeing a pointer against the wrong device's counter).
pub struct MultiDeviceSimulatedLimiter {
    devices: Vec<SimulatedLimiter>,
    /// Maps external pointer -> (device_idx, device-local pointer) for correct free routing.
    /// We use a separate external pointer space to avoid collisions between devices
    /// (each per-device SimulatedLimiter has its own pointer counter starting at 1).
    pointer_device_map: DashMap<usize, (usize, usize)>,
    /// Shared pointer counter across all devices to generate unique external pointers.
    next_pointer: AtomicUsize,
}

impl MultiDeviceSimulatedLimiter {
    pub fn new(device_limits: &[u64]) -> Self {
        Self {
            devices: device_limits.iter().map(|&limit| SimulatedLimiter::new(limit)).collect(),
            pointer_device_map: DashMap::new(),
            next_pointer: AtomicUsize::new(1),
        }
    }

    pub fn try_alloc(&self, device_idx: usize, size: u64) -> Result<usize, ()> {
        let device = self.devices.get(device_idx).ok_or(())?;
        let device_pointer = device.try_alloc(size)?;
        Ok(self.register_pointer(device_idx, device_pointer, size))
    }

    /// Map a device-local pointer to a unique external pointer and track it.
    fn register_pointer(&self, device_idx: usize, device_pointer: usize, size: u64) -> usize {
        let external_pointer = self.next_pointer.fetch_add(1, Ordering::Relaxed);
        if size > 0 {
            self.pointer_device_map.insert(external_pointer, (device_idx, device_pointer));
        }
        external_pointer
    }

    pub fn free(&self, pointer: usize) -> bool {
        let Some((_, (device_idx, device_pointer))) = self.pointer_device_map.remove(&pointer) else {
            return false;
        };
        self.devices[device_idx].free(device_pointer)
    }

    /// Drain all tracked allocations across all devices.
    ///
    /// Models the real atexit handler: clears the pointer-to-device map, then
    /// delegates to each per-device `SimulatedLimiter::drain_allocations()` which
    /// clears its own tracker and does a bulk `saturating_sub_pod_memory_used`.
    /// Returns total bytes drained across all devices.
    pub fn drain_allocations(&self) -> u64 {
        // Clear the multi-device routing map (entries are no longer needed
        // because per-device trackers will be drained independently).
        self.pointer_device_map.retain(|_, _| false);

        // Delegate to each device's drain — mirrors the real limiter's
        // per-device aggregation + bulk saturating_sub pattern.
        self.devices.iter().map(|device| device.drain_allocations()).sum()
    }

    pub fn pod_memory_used(&self, device_idx: usize) -> u64 {
        self.devices[device_idx].pod_memory_used()
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn proc_usage(&self, device_idx: usize) -> u64 {
        self.devices[device_idx].proc_usage()
    }

    /// Inject stale usage on a specific device (models dead process).
    pub fn inject_stale_usage(&self, device_idx: usize, size: u64) {
        self.devices[device_idx].inject_stale_usage(size);
    }

    /// Recover stale usage on a specific device.
    pub fn recover_stale_usage(&self, device_idx: usize, size: u64) {
        self.devices[device_idx].recover_stale_usage(size);
    }

    /// Attempt allocation with reap-on-OOM retry on a specific device.
    pub fn try_alloc_with_reap(
        &self,
        device_idx: usize,
        size: u64,
        reap_fn: impl FnOnce(&Self) -> u64,
    ) -> Result<usize, ()> {
        let device = self.devices.get(device_idx).ok_or(())?;
        match device.try_alloc(size) {
            Ok(device_pointer) => Ok(self.register_pointer(device_idx, device_pointer, size)),
            Err(()) => {
                let recovered = reap_fn(self);
                if recovered > 0 {
                    let device_pointer = device.try_alloc(size)?;
                    Ok(self.register_pointer(device_idx, device_pointer, size))
                } else {
                    Err(())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_alloc_free_cycle() {
        let limiter = SimulatedLimiter::new(1024);
        let pointer = limiter.try_alloc(512).expect("should succeed");
        assert_eq!(limiter.pod_memory_used(), 512);
        assert!(limiter.free(pointer));
        assert_eq!(limiter.pod_memory_used(), 0);
    }

    /// Zero-size allocations succeed, return a unique pointer, but are not
    /// tracked — free returns false and no accounting is affected.
    /// This matches real HIP behavior where hipMalloc(&ptr, 0) succeeds.
    #[test]
    fn zero_size_alloc_not_tracked() {
        let limiter = SimulatedLimiter::new(1024);

        let ptr1 = limiter.try_alloc(0).expect("zero-size should succeed");
        let ptr2 = limiter.try_alloc(0).expect("zero-size should succeed");
        assert_ne!(ptr1, ptr2, "each zero-size alloc gets a unique pointer");

        assert_eq!(limiter.pod_memory_used(), 0, "zero-size must not affect counter");
        assert_eq!(limiter.proc_usage(), 0, "zero-size must not affect proc_usage");
        assert_eq!(limiter.allocation_count(), 0, "zero-size must not be tracked");

        assert!(!limiter.free(ptr1), "free of zero-size pointer returns false");
        assert!(!limiter.free(ptr2), "free of zero-size pointer returns false");
        assert_eq!(limiter.pod_memory_used(), 0, "free of untracked pointer is noop");
    }
}
