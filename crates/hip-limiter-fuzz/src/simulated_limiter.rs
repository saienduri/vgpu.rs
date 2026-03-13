use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use dashmap::DashMap;

/// Maximum single allocation size — rejects before fetch_add to prevent
/// transient wrapping of the atomic counter.
const MAX_ALLOC_SIZE: u64 = u64::MAX / 2;

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
            self.pod_memory_used.fetch_sub(size, Ordering::AcqRel);
            return Err(());
        }

        let pointer = self.next_pointer.fetch_add(1, Ordering::Relaxed);
        self.allocation_tracker.insert(pointer, size);
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
        // Use saturating_sub via CAS loop to match the real limiter's
        // saturating_fetch_sub_pod_memory_used (prevents underflow wrapping).
        loop {
            let current = self.pod_memory_used.load(Ordering::Acquire);
            let new_value = current.saturating_sub(size);
            if self.pod_memory_used.compare_exchange_weak(
                current, new_value, Ordering::AcqRel, Ordering::Acquire,
            ).is_ok() {
                break;
            }
        }
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

    /// Returns the number of live tracked allocations.
    pub fn allocation_count(&self) -> usize {
        self.allocation_tracker.len()
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
            self.pod_memory_used.fetch_sub(estimated_size, Ordering::AcqRel);
            return Err(());
        }

        // Phase 2: Native allocator
        if !native_succeeds {
            self.pod_memory_used.fetch_sub(estimated_size, Ordering::AcqRel);
            return Err(());
        }

        // Phase 3: Reserve alignment overhead (extra = actual_size - estimated_size)
        let extra = actual_size.saturating_sub(estimated_size);

        if extra > 0 {
            let prev = self.pod_memory_used.fetch_add(extra, Ordering::AcqRel);
            let new_total = prev.saturating_add(extra);

            if new_total > self.mem_limit {
                // Extra overhead pushes over limit — rollback everything
                self.pod_memory_used.fetch_sub(extra, Ordering::AcqRel);
                self.pod_memory_used.fetch_sub(estimated_size, Ordering::AcqRel);
                return Err(());
            }
        }

        // Phase 4: Record with actual_size
        let pointer = self.next_pointer.fetch_add(1, Ordering::Relaxed);
        self.allocation_tracker.insert(pointer, actual_size);
        Ok(pointer)
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
            self.pod_memory_used.fetch_sub(size, Ordering::AcqRel);
            return Err(());
        }

        // Simulate native failure — roll back the reservation
        self.pod_memory_used.fetch_sub(size, Ordering::AcqRel);
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
        let external_pointer = self.next_pointer.fetch_add(1, Ordering::Relaxed);
        if size > 0 {
            self.pointer_device_map.insert(external_pointer, (device_idx, device_pointer));
        }
        Ok(external_pointer)
    }

    pub fn free(&self, pointer: usize) -> bool {
        let Some((_, (device_idx, device_pointer))) = self.pointer_device_map.remove(&pointer) else {
            return false;
        };
        self.devices[device_idx].free(device_pointer)
    }

    pub fn pod_memory_used(&self, device_idx: usize) -> u64 {
        self.devices[device_idx].pod_memory_used()
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
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
}
