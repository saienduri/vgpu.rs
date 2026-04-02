//! Multi-process pod simulation for fuzzing the limiter's lifecycle behaviors.
//!
//! Models a pod with N GPU devices and M concurrent processes sharing a
//! [`ProcSlotTable`]. Each process has its own allocation tracker and injected
//! non-hipMalloc overhead, enabling tests for:
//! - Stale proc slot cleanup during reconciliation
//! - Effective mem_limit tightening under overhead pressure
//! - Process lifecycle (spawn, SIGKILL, clean drain)
//! - Slot exhaustion and reaping
//!
//! DRM fdinfo is simulated by injecting `non_hip` values directly — no GPU needed.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use dashmap::DashMap;
use utils::shared_memory::proc_slots::ProcSlotTable;

use crate::saturating_fetch_sub;

/// Maximum single allocation size — mirrors real limiter guard.
const MAX_ALLOC_SIZE: u64 = u64::MAX / 2;

// ---------------------------------------------------------------------------
// DeviceState — per-GPU shared counters (mirrors SharedDeviceInfoV2)
// ---------------------------------------------------------------------------

pub struct DeviceState {
    mem_limit: u64,
    pod_memory_used: AtomicU64,
    effective_mem_limit: AtomicU64,
}

impl DeviceState {
    fn new(mem_limit: u64) -> Self {
        Self {
            mem_limit,
            pod_memory_used: AtomicU64::new(0),
            effective_mem_limit: AtomicU64::new(0),
        }
    }

    fn active_limit(&self) -> u64 {
        let effective = self.effective_mem_limit.load(Ordering::Acquire);
        if effective > 0 { effective } else { self.mem_limit }
    }
}

// ---------------------------------------------------------------------------
// SimulatedProcess — per-process state (mirrors Limiter struct fields)
// ---------------------------------------------------------------------------

pub struct SimulatedProcess {
    pub pid: u32,
    pub slot_idx: Option<usize>,
    alloc_tracker: DashMap<usize, (usize, u64)>, // ptr -> (device_idx, size)
    alloc_count: AtomicU64,
    /// Per-device non-hipMalloc overhead (injected, simulates DRM fdinfo measurement).
    non_hip: Vec<u64>,
    next_pointer: AtomicUsize,
    alive: bool,
}

impl SimulatedProcess {
    fn new(pid: u32, slot_idx: Option<usize>, num_devices: usize, non_hip: &[u64]) -> Self {
        let mut non_hip_vec = non_hip.to_vec();
        non_hip_vec.resize(num_devices, 0);
        Self {
            pid,
            slot_idx,
            alloc_tracker: DashMap::new(),
            alloc_count: AtomicU64::new(0),
            non_hip: non_hip_vec,
            next_pointer: AtomicUsize::new(1),
            alive: true,
        }
    }

    pub fn is_alive(&self) -> bool {
        self.alive
    }

    pub fn alloc_count(&self) -> u64 {
        self.alloc_count.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// SimulatedPod — the pod orchestrator
// ---------------------------------------------------------------------------

pub struct SimulatedPod {
    pub devices: Vec<DeviceState>,
    proc_slots: Box<ProcSlotTable>,
    processes: Vec<SimulatedProcess>,
    next_pid: u32,
}

impl SimulatedPod {
    /// Create a new pod with the given per-device memory limits.
    pub fn new(device_limits: &[u64]) -> Self {
        let proc_slots = ProcSlotTable::new_zeroed();
        proc_slots.initialize();
        Self {
            devices: device_limits.iter().map(|&limit| DeviceState::new(limit)).collect(),
            proc_slots,
            processes: Vec::new(),
            // Use high PIDs that `is_process_alive` will report as dead when the
            // process is "killed" (no real OS process exists for these PIDs).
            next_pid: 4_000_000,
        }
    }

    /// Spawn a new process with the given per-device non-hipMalloc overhead.
    /// Claims a proc slot and writes the initial non_hip values.
    /// Returns the process index in `self.processes`.
    pub fn spawn_process(&mut self, non_hip_per_device: &[u64]) -> usize {
        let pid = self.next_pid;
        self.next_pid += 1;

        let slot_idx = self.proc_slots.claim_slot(pid);
        let num_devices = self.devices.len();
        let process = SimulatedProcess::new(pid, slot_idx, num_devices, non_hip_per_device);

        // Write initial non_hip to proc slot.
        if let Some(idx) = slot_idx {
            for (dev, &overhead) in non_hip_per_device.iter().enumerate() {
                if dev < num_devices {
                    self.proc_slots.write_non_hip(idx, dev, overhead);
                }
            }
        }

        let proc_idx = self.processes.len();
        self.processes.push(process);
        proc_idx
    }

    /// Kill a process without draining (simulates SIGKILL).
    /// The proc slot retains its PID + counters as stale data.
    pub fn kill_process(&mut self, proc_idx: usize) {
        self.processes[proc_idx].alive = false;
        // Intentionally do NOT release the proc slot or zero counters.
        // The slot will be reaped by a future reconciliation.
    }

    /// Cleanly drain a process (simulates normal exit with atexit handler).
    /// Subtracts all tracked allocations from pod_memory_used and releases the proc slot.
    pub fn drain_process(&mut self, proc_idx: usize) -> u64 {
        let process = &mut self.processes[proc_idx];
        process.alive = false;

        // Aggregate per-device totals from the tracker.
        let mut per_device_total = vec![0u64; self.devices.len()];
        process.alloc_tracker.retain(|_, (dev_idx, size)| {
            if *dev_idx < per_device_total.len() {
                per_device_total[*dev_idx] += *size;
            }
            false // remove all entries
        });

        let mut total_drained = 0u64;
        for (dev_idx, &amount) in per_device_total.iter().enumerate() {
            if amount > 0 {
                saturating_fetch_sub(&self.devices[dev_idx].pod_memory_used, amount);
                total_drained += amount;
            }
        }

        // Release the proc slot (zeros counters + non_hip, sets PID to 0).
        if let Some(slot_idx) = process.slot_idx {
            self.proc_slots.zero_and_release(slot_idx);
        }

        total_drained
    }

    /// Attempt an allocation for a process on a device.
    /// Uses reserve-then-allocate with the device's active limit.
    /// Triggers reconciliation every 100 allocs.
    pub fn try_alloc(
        &self,
        proc_idx: usize,
        device_idx: usize,
        size: u64,
    ) -> Result<usize, ()> {
        let process = &self.processes[proc_idx];
        assert!(process.alive, "cannot alloc on dead process");

        if size == 0 {
            let ptr = process.next_pointer.fetch_add(1, Ordering::Relaxed);
            return Ok(ptr);
        }
        if size > MAX_ALLOC_SIZE {
            return Err(());
        }

        let device = &self.devices[device_idx];

        // Reserve-then-allocate.
        let previous = device.pod_memory_used.fetch_add(size, Ordering::AcqRel);
        let new_used = previous.saturating_add(size);

        if new_used > device.active_limit() {
            saturating_fetch_sub(&device.pod_memory_used, size);
            return Err(());
        }

        let ptr = process.next_pointer.fetch_add(1, Ordering::Relaxed);
        process.alloc_tracker.insert(ptr, (device_idx, size));

        // Update proc slot usage.
        if let Some(slot_idx) = process.slot_idx {
            self.proc_slots.add_usage(slot_idx, device_idx, size);
        }

        // Trigger reconciliation every 100 allocs (matches real limiter: count % 100 == 0).
        let count = process.alloc_count.fetch_add(1, Ordering::Relaxed);
        if count % 100 == 0 {
            self.reconcile(proc_idx);
        }

        Ok(ptr)
    }

    /// Free a previously allocated pointer.
    pub fn free(&self, proc_idx: usize, pointer: usize) -> bool {
        let process = &self.processes[proc_idx];
        debug_assert!(process.alive, "cannot free on dead process");
        let Some((_, (device_idx, size))) = process.alloc_tracker.remove(&pointer) else {
            return false;
        };
        saturating_fetch_sub(&self.devices[device_idx].pod_memory_used, size);
        if let Some(slot_idx) = process.slot_idx {
            self.proc_slots.sub_usage(slot_idx, device_idx, size);
        }
        true
    }

    /// Reconciliation: reap dead processes, write non_hip, recalculate effective limits.
    /// Mirrors the real `limiter.rs` reconciliation path.
    pub fn reconcile(&self, proc_idx: usize) {
        let process = &self.processes[proc_idx];
        assert!(process.alive, "cannot reconcile dead process");

        // 1. Reap dead processes from proc slots.
        let reaped = self.reap_dead_pids();

        // Subtract reaped usage from pod_memory_used (mirrors real reap_dead_pids).
        for (_dead_pid, usage) in &reaped {
            for (dev_idx, &bytes) in usage.iter().enumerate() {
                if bytes > 0 && dev_idx < self.devices.len() {
                    saturating_fetch_sub(&self.devices[dev_idx].pod_memory_used, bytes);
                }
            }
        }

        // 2. Write our non_hip to proc slot for each device.
        if let Some(slot_idx) = process.slot_idx {
            for (dev_idx, &overhead) in process.non_hip.iter().enumerate() {
                if dev_idx < self.devices.len() {
                    self.proc_slots.write_non_hip(slot_idx, dev_idx, overhead);
                }
            }
        }

        // 3. Recalculate effective_mem_limit for all devices.
        for (dev_idx, device) in self.devices.iter().enumerate() {
            let total_overhead = self.proc_slots.sum_non_hip_for_device(dev_idx);
            let effective = device.mem_limit.saturating_sub(total_overhead);
            device.effective_mem_limit.store(effective, Ordering::Release);
        }
    }

    /// Reap dead PIDs from the proc slot table.
    ///
    /// We can't use `ProcSlotTable::find_dead_slots()` because it checks OS liveness
    /// via `kill(pid, 0)`, and our simulated PIDs (4_000_000+) have no real OS process.
    /// Instead, we scan slots and check our own process list for liveness.
    fn reap_dead_pids(&self) -> Vec<(u32, [u64; utils::shared_memory::MAX_DEVICES])> {
        // Build a set of live PIDs.
        let live_pids: std::collections::HashSet<u32> = self
            .processes
            .iter()
            .filter(|p| p.alive)
            .map(|p| p.pid)
            .collect();

        // Scan slots for PIDs not in the live set.
        let mut dead = Vec::new();
        for (i, slot) in self.proc_slots.slots.iter().enumerate() {
            let pid = slot.pid.load(Ordering::Acquire);
            if pid != 0 && !live_pids.contains(&pid) {
                dead.push((i, pid));
            }
        }

        let mut reaped = Vec::with_capacity(dead.len());
        for (slot_idx, dead_pid) in dead {
            if let Some(usage) = self.proc_slots.try_claim_and_release(slot_idx, dead_pid) {
                reaped.push((dead_pid, usage));
            }
        }
        reaped
    }

    // -- Accessors --

    pub fn pod_memory_used(&self, device_idx: usize) -> u64 {
        self.devices[device_idx].pod_memory_used.load(Ordering::Acquire)
    }

    pub fn effective_mem_limit(&self, device_idx: usize) -> u64 {
        self.devices[device_idx].effective_mem_limit.load(Ordering::Acquire)
    }

    pub fn process(&self, proc_idx: usize) -> &SimulatedProcess {
        &self.processes[proc_idx]
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn sum_non_hip_for_device(&self, device_idx: usize) -> u64 {
        self.proc_slots.sum_non_hip_for_device(device_idx)
    }

    pub fn live_process_count(&self) -> usize {
        self.processes.iter().filter(|p| p.alive).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn basic_spawn_alloc_drain() {
        let mut pod = SimulatedPod::new(&[128 * GIB]);
        let p0 = pod.spawn_process(&[0]);

        let ptr = pod.try_alloc(p0, 0, 1024).unwrap();
        assert_eq!(pod.pod_memory_used(0), 1024);

        assert!(pod.free(p0, ptr));
        assert_eq!(pod.pod_memory_used(0), 0);

        pod.drain_process(p0);
        assert!(!pod.process(p0).is_alive());
    }

    #[test]
    fn kill_leaves_stale_slot() {
        let mut pod = SimulatedPod::new(&[128 * GIB]);
        let p0 = pod.spawn_process(&[5 * GIB]);

        // Allocate some memory.
        let _ptr = pod.try_alloc(p0, 0, 1000).unwrap();

        // Kill without drain.
        pod.kill_process(p0);

        // non_hip is still in the slot (stale).
        assert_eq!(pod.sum_non_hip_for_device(0), 5 * GIB);
        // pod_memory_used is still elevated.
        assert_eq!(pod.pod_memory_used(0), 1000);
    }

    #[test]
    fn reconciliation_reaps_stale_non_hip() {
        let mut pod = SimulatedPod::new(&[128 * GIB]);

        // Spawn and kill 3 processes with 13 GiB non_hip each.
        for _ in 0..3 {
            let p = pod.spawn_process(&[13 * GIB]);
            pod.kill_process(p);
        }

        // Total non_hip = 3 * 13 = 39 GiB (stale).
        assert_eq!(pod.sum_non_hip_for_device(0), 39 * GIB);

        // Spawn a live process.
        let live = pod.spawn_process(&[2 * GIB]);

        // Reconcile — should reap dead slots.
        pod.reconcile(live);

        // After reap: only live process's non_hip (2 GiB) remains.
        assert_eq!(pod.sum_non_hip_for_device(0), 2 * GIB);

        // Effective limit should be mem_limit - 2 GiB.
        assert_eq!(pod.effective_mem_limit(0), 128 * GIB - 2 * GIB);
    }

    #[test]
    fn effective_limit_gates_allocation() {
        let mut pod = SimulatedPod::new(&[128 * GIB]);
        let p0 = pod.spawn_process(&[100 * GIB]); // heavy overhead

        // Reconcile to set effective limit = 128 - 100 = 28 GiB.
        pod.reconcile(p0);
        assert_eq!(pod.effective_mem_limit(0), 28 * GIB);

        // Alloc 20 GiB — should succeed.
        let ptr = pod.try_alloc(p0, 0, 20 * GIB).unwrap();

        // Alloc 10 GiB — should be denied (20 + 10 = 30 > 28).
        assert!(pod.try_alloc(p0, 0, 10 * GIB).is_err());

        // Free and retry.
        pod.free(p0, ptr);
        assert!(pod.try_alloc(p0, 0, 10 * GIB).is_ok());
    }

    #[test]
    fn multi_device_overhead_independence() {
        let mut pod = SimulatedPod::new(&[128 * GIB, 128 * GIB]);

        // Process with overhead only on device 0.
        let p0 = pod.spawn_process(&[50 * GIB, 0]);
        pod.reconcile(p0);

        // Device 0: effective = 78 GiB.
        assert_eq!(pod.effective_mem_limit(0), 78 * GIB);
        // Device 1: effective = 128 GiB (no overhead).
        assert_eq!(pod.effective_mem_limit(1), 128 * GIB);

        // Large alloc succeeds on device 1 but not on device 0.
        assert!(pod.try_alloc(p0, 0, 80 * GIB).is_err());
        assert!(pod.try_alloc(p0, 1, 80 * GIB).is_ok());
    }
}
