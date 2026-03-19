use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use anyhow::Context;
use shared_memory::{Shmem, ShmemConf, ShmemError};
use tracing::info;

use super::MAX_DEVICES;

pub const MAX_PROC_SLOTS: usize = 128;
const PROC_SLOTS_MAGIC: u32 = 0x50524F43; // "PROC"
const PROC_SLOTS_SHM_ID: &str = "proc_slots";

// ---------------------------------------------------------------------------
// ProcSlot / ProcSlotTable — lock-free per-PID per-device usage tracking
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct ProcSlot {
    pub pid: AtomicU32, // 0 = empty slot
    _pad: u32,
    pub used: [AtomicU64; MAX_DEVICES], // per-device memory usage
}

#[repr(C)]
pub struct ProcSlotTable {
    pub magic: AtomicU32,
    _reserved: u32,
    pub slots: [ProcSlot; MAX_PROC_SLOTS],
}

impl ProcSlotTable {
    /// Returns `true` if the magic number has been written.
    pub fn is_initialized(&self) -> bool {
        self.magic.load(Ordering::Acquire) == PROC_SLOTS_MAGIC
    }

    /// Write the magic number (idempotent — safe to call twice).
    pub fn initialize(&self) {
        self.magic.store(PROC_SLOTS_MAGIC, Ordering::Release);
    }

    /// Claim the first empty slot for `pid`.  Returns the slot index on success.
    /// After a successful CAS the per-device counters are zeroed so stale data
    /// from a previous occupant can never leak.
    pub fn claim_slot(&self, pid: u32) -> Option<usize> {
        for (i, slot) in self.slots.iter().enumerate() {
            if slot
                .pid
                .compare_exchange(0, pid, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                // Zero the usage counters (previous occupant may have left data).
                for counter in &slot.used {
                    counter.store(0, Ordering::Release);
                }
                return Some(i);
            }
        }
        None
    }

    /// Release a slot (mark it empty).
    pub fn release_slot(&self, slot_idx: usize) {
        if slot_idx < MAX_PROC_SLOTS {
            self.slots[slot_idx].pid.store(0, Ordering::Release);
        }
    }

    /// Check whether a process is still alive via `kill(pid, 0)`.
    /// EPERM means the process exists but we lack permission — treat as alive.
    pub fn is_process_alive(pid: u32) -> bool {
        if pid == 0 {
            return false;
        }
        #[cfg(unix)]
        {
            let ret = unsafe { libc::kill(pid as i32, 0) };
            if ret == 0 {
                return true;
            }
            // EPERM → process exists, we just can't signal it
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            errno == libc::EPERM
        }
        #[cfg(not(unix))]
        {
            true // conservative fallback
        }
    }

    /// Scan all slots and return `(slot_idx, pid)` for every slot whose PID
    /// is no longer alive. The caller must pass the returned PID (not a re-read)
    /// to `try_claim_and_release` to avoid a TOCTOU race where a new process
    /// claims the slot between the scan and the reap CAS.
    pub fn find_dead_slots(&self) -> Vec<(usize, u32)> {
        let mut dead = Vec::new();
        for (i, slot) in self.slots.iter().enumerate() {
            let pid = slot.pid.load(Ordering::Acquire);
            if pid != 0 && !Self::is_process_alive(pid) {
                dead.push((i, pid));
            }
        }
        dead
    }

    /// Atomically add `size` bytes to a device counter.
    pub fn add_usage(&self, slot_idx: usize, device_idx: usize, size: u64) {
        if slot_idx < MAX_PROC_SLOTS && device_idx < MAX_DEVICES {
            self.slots[slot_idx].used[device_idx].fetch_add(size, Ordering::Relaxed);
        }
    }

    /// Atomically subtract `size` bytes from a device counter (saturating).
    /// Uses a CAS loop so we never wrap below zero.
    pub fn sub_usage(&self, slot_idx: usize, device_idx: usize, size: u64) {
        if slot_idx >= MAX_PROC_SLOTS || device_idx >= MAX_DEVICES {
            return;
        }
        let counter = &self.slots[slot_idx].used[device_idx];
        loop {
            let current = counter.load(Ordering::Relaxed);
            let new = current.saturating_sub(size);
            if counter
                .compare_exchange_weak(current, new, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Read all per-device usage values for a slot.
    pub fn read_slot_usage(&self, slot_idx: usize) -> [u64; MAX_DEVICES] {
        let mut out = [0u64; MAX_DEVICES];
        if slot_idx < MAX_PROC_SLOTS {
            for (i, counter) in self.slots[slot_idx].used.iter().enumerate() {
                out[i] = counter.load(Ordering::Relaxed);
            }
        }
        out
    }

    /// Zero all per-device usage counters for a slot, then release it.
    ///
    /// The ordering matters: counters must be zeroed BEFORE the PID is set to 0,
    /// so a concurrent `claim_slot` never sees stale usage data from the previous
    /// occupant (claim_slot also zeroes, but this prevents a brief window where
    /// the PID is 0 but counters still hold old values).
    pub fn zero_and_release(&self, slot_idx: usize) {
        if slot_idx < MAX_PROC_SLOTS {
            for counter in &self.slots[slot_idx].used {
                counter.store(0, Ordering::Release);
            }
            self.release_slot(slot_idx);
        }
    }

    /// Atomically claim the right to reap a slot by CAS'ing its PID to 0.
    ///
    /// Only the process that successfully CAS's `expected_pid → 0` may subtract
    /// the slot's usage from `pod_memory_used`. This prevents double-subtraction
    /// when two processes call `reap_dead` concurrently for the same dead PID.
    ///
    /// Usage is read BEFORE the CAS to prevent a concurrent `claim_slot` from
    /// zeroing counters between our CAS and our read. Once the CAS sets PID to 0,
    /// the slot is "free" and a new process may claim it immediately.
    ///
    /// Returns the per-device usage snapshot on success, or `None` if another
    /// process already reaped this slot.
    pub fn try_claim_and_release(&self, slot_idx: usize, expected_pid: u32) -> Option<[u64; MAX_DEVICES]> {
        if slot_idx >= MAX_PROC_SLOTS || expected_pid == 0 {
            return None;
        }
        // Snapshot usage BEFORE releasing the slot. The dead process is no longer
        // writing, so these values are final. Reading after CAS would race with
        // a concurrent claim_slot that zeroes counters on the newly-freed slot.
        let usage = self.read_slot_usage(slot_idx);

        // Atomically claim reap rights: only the winner gets to subtract usage.
        if self.slots[slot_idx]
            .pid
            .compare_exchange(expected_pid, 0, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return None; // Another process already reaped this slot.
        }
        // CAS succeeded — we own this slot. Zero counters before any new claimer
        // can observe them (defense-in-depth; claim_slot also zeroes).
        for counter in &self.slots[slot_idx].used {
            counter.store(0, Ordering::Release);
        }
        Some(usage)
    }

    /// Create a heap-allocated zeroed `ProcSlotTable` (for tests).
    ///
    /// # Safety
    /// `ProcSlotTable` is `#[repr(C)]` with all-atomic fields. Zero is a valid
    /// bit pattern for atomics on all platforms Rust targets (AtomicU32(0),
    /// AtomicU64(0)). This mirrors how the table appears in freshly-mmap'd SHM
    /// (the kernel zero-fills new pages).
    pub fn new_zeroed() -> Box<ProcSlotTable> {
        // SAFETY: see doc comment above.
        unsafe { Box::new(std::mem::zeroed()) }
    }
}

// ---------------------------------------------------------------------------
// ProcSlotHandle — mmap wrapper with automatic slot lifecycle
// ---------------------------------------------------------------------------

/// Sentinel value for `slot_idx` meaning "no slot claimed".
const NO_SLOT: usize = usize::MAX;

pub struct ProcSlotHandle {
    _shmem: Shmem, // held for drop (keeps mapping alive)
    table: *const ProcSlotTable,
    /// Our claimed slot index, or `NO_SLOT` if none (slot exhaustion or post-drain).
    /// AtomicUsize so that concurrent threads can safely read without data races.
    slot_idx: AtomicUsize,
}

// SAFETY: ProcSlotTable uses only atomic fields, safe for cross-process/thread access.
// The `*const ProcSlotTable` pointer is valid for the lifetime of `_shmem` (which owns
// the memory mapping). `slot_idx` is AtomicUsize, safe for concurrent access.
unsafe impl Send for ProcSlotHandle {}
unsafe impl Sync for ProcSlotHandle {}

impl ProcSlotHandle {
    /// Create (or open) the proc-slots SHM segment, initialise if fresh, and
    /// claim a slot for the current process.
    pub fn create_and_claim(shm_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        std::fs::create_dir_all(shm_dir.as_ref())?;

        let (mut shmem, created_fresh) = match ShmemConf::new()
            .size(std::mem::size_of::<ProcSlotTable>())
            .use_tmpfs_with_dir(shm_dir.as_ref())
            .os_id(PROC_SLOTS_SHM_ID)
            .create()
        {
            Ok(shmem) => (shmem, true),
            Err(ShmemError::LinkExists) | Err(ShmemError::MappingIdExists) => {
                let shmem = ShmemConf::new()
                    .size(std::mem::size_of::<ProcSlotTable>())
                    .use_tmpfs_with_dir(shm_dir.as_ref())
                    .os_id(PROC_SLOTS_SHM_ID)
                    .open()
                    .context("Failed to open existing proc_slots SHM")?;
                (shmem, false)
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Failed to create proc_slots SHM: {e}"));
            }
        };

        shmem.set_owner(false);

        let table = shmem.as_ptr() as *const ProcSlotTable;
        let table_ref = unsafe { &*table };

        if created_fresh {
            table_ref.initialize();
            info!(path = ?shm_dir.as_ref(), "Created proc_slots SHM segment");
        } else if !table_ref.is_initialized() {
            // Opened existing segment that hasn't been initialized yet (race).
            table_ref.initialize();
        }

        let pid = std::process::id();
        let slot_idx = table_ref.claim_slot(pid);
        if let Some(idx) = slot_idx {
            info!(pid, slot_idx = idx, "Claimed proc slot");
        } else {
            tracing::warn!(pid, "All proc slots full — per-PID tracking disabled");
        }

        Ok(Self {
            _shmem: shmem,
            table,
            slot_idx: AtomicUsize::new(slot_idx.unwrap_or(NO_SLOT)),
        })
    }

    /// Access the underlying table.
    pub fn table(&self) -> &ProcSlotTable {
        unsafe { &*self.table }
    }

    /// Our claimed slot index (if any).
    pub fn slot_idx(&self) -> Option<usize> {
        match self.slot_idx.load(Ordering::Acquire) {
            NO_SLOT => None,
            idx => Some(idx),
        }
    }

    /// Record an allocation on `device_idx`.
    pub fn add_usage(&self, device_idx: usize, size: u64) {
        let idx = self.slot_idx.load(Ordering::Acquire);
        if idx != NO_SLOT {
            self.table().add_usage(idx, device_idx, size);
        }
    }

    /// Record a free on `device_idx` (saturating).
    pub fn sub_usage(&self, device_idx: usize, size: u64) {
        let idx = self.slot_idx.load(Ordering::Acquire);
        if idx != NO_SLOT {
            self.table().sub_usage(idx, device_idx, size);
        }
    }

    /// Scan for dead processes, atomically claim their slots, and return usage.
    ///
    /// Uses CAS on each dead slot's PID field to prevent double-reaping: only the
    /// process that successfully CAS's `dead_pid → 0` subtracts usage. This is safe
    /// under concurrent `reap_dead` calls from multiple processes.
    ///
    /// The PID observed by `find_dead_slots` is passed directly to `try_claim_and_release`
    /// (never re-read from the slot). Re-reading would create a TOCTOU race: another
    /// reaper could release the slot, a new process could claim it, and the re-read
    /// would return the new (live) PID — causing the CAS to evict the live process.
    pub fn reap_dead(&self) -> Vec<(u32, [u64; MAX_DEVICES])> {
        let table = self.table();
        let dead = table.find_dead_slots();
        let mut reaped = Vec::with_capacity(dead.len());

        for (slot_idx, dead_pid) in dead {
            if let Some(usage) = table.try_claim_and_release(slot_idx, dead_pid) {
                reaped.push((dead_pid, usage));
                tracing::info!(dead_pid, slot_idx, "Reaped dead process slot");
            }
        }
        reaped
    }

    /// Read our own slot's usage, zero it, release the slot, and invalidate
    /// our slot index so subsequent `add_usage`/`sub_usage` calls are no-ops.
    ///
    /// Intended for atexit / drain_allocations. After this call, `slot_idx()`
    /// returns `None`.
    pub fn drain_our_slot(&self) -> Option<[u64; MAX_DEVICES]> {
        // Atomically invalidate slot_idx FIRST so concurrent add_usage/sub_usage
        // become no-ops before we release the slot. Without this, a concurrent
        // thread could write into a freed slot between zero_and_release and
        // the slot_idx invalidation.
        let idx = self.slot_idx.swap(NO_SLOT, Ordering::AcqRel);
        if idx == NO_SLOT {
            return None;
        }
        let table = self.table();
        let usage = table.read_slot_usage(idx);
        table.zero_and_release(idx);
        Some(usage)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // -- ProcSlotTable unit tests --

    #[test]
    fn claim_and_release() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();

        let idx = table.claim_slot(42).expect("should claim");
        assert_eq!(table.slots[idx].pid.load(Ordering::Relaxed), 42);

        table.release_slot(idx);
        assert_eq!(table.slots[idx].pid.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn claim_fills_up() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();

        for i in 0..MAX_PROC_SLOTS {
            assert!(
                table.claim_slot((i + 1) as u32).is_some(),
                "slot {i} should be claimable"
            );
        }
        assert!(table.claim_slot(9999).is_none(), "table should be full");
    }

    #[test]
    fn add_sub_usage() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();
        let idx = table.claim_slot(1).unwrap();

        table.add_usage(idx, 0, 1000);
        table.add_usage(idx, 0, 500);
        assert_eq!(table.read_slot_usage(idx)[0], 1500);

        table.sub_usage(idx, 0, 400);
        assert_eq!(table.read_slot_usage(idx)[0], 1100);
    }

    #[test]
    fn sub_usage_saturates() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();
        let idx = table.claim_slot(1).unwrap();

        table.add_usage(idx, 0, 100);
        table.sub_usage(idx, 0, 200);
        assert_eq!(table.read_slot_usage(idx)[0], 0, "should saturate to 0");
    }

    #[test]
    fn read_slot_usage_multi_device() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();
        let idx = table.claim_slot(1).unwrap();

        table.add_usage(idx, 0, 1000);
        table.add_usage(idx, 2, 2000);

        let usage = table.read_slot_usage(idx);
        assert_eq!(usage[0], 1000);
        assert_eq!(usage[1], 0);
        assert_eq!(usage[2], 2000);
    }

    #[test]
    fn find_dead_slots_detects_current_process() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();

        let pid = std::process::id();
        let idx = table.claim_slot(pid).unwrap();
        table.add_usage(idx, 0, 500);

        let dead = table.find_dead_slots();
        assert!(
            dead.is_empty(),
            "current process should be alive, got {dead:?}"
        );
    }

    #[test]
    fn find_dead_slots_detects_fake_pid() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();

        // PID 4_000_000 almost certainly doesn't exist.
        let idx = table.claim_slot(4_000_000).unwrap();
        table.add_usage(idx, 0, 999);

        let dead = table.find_dead_slots();
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0].0, idx);
        assert_eq!(dead[0].1, 4_000_000); // returns (slot_idx, pid)
    }

    #[test]
    fn claim_reuses_released_slot() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();

        let idx1 = table.claim_slot(1).unwrap();
        table.add_usage(idx1, 0, 5000);
        table.add_usage(idx1, 1, 3000);
        table.release_slot(idx1);

        let idx2 = table.claim_slot(2).unwrap();
        assert_eq!(idx1, idx2, "should reuse the first released slot");

        // claim_slot zeroes counters — stale data must never leak to new occupant.
        let usage = table.read_slot_usage(idx2);
        assert_eq!(usage[0], 0, "device 0 counter must be zeroed after reclaim");
        assert_eq!(usage[1], 0, "device 1 counter must be zeroed after reclaim");
    }

    #[test]
    fn try_claim_and_release_prevents_double_reap() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();

        let idx = table.claim_slot(4_000_000).unwrap();
        table.add_usage(idx, 0, 5000);

        // First claim succeeds.
        let usage = table.try_claim_and_release(idx, 4_000_000);
        assert!(usage.is_some());
        assert_eq!(usage.unwrap()[0], 5000);

        // Second claim fails — slot already reaped.
        let usage2 = table.try_claim_and_release(idx, 4_000_000);
        assert!(usage2.is_none());
    }

    #[test]
    fn add_usage_out_of_bounds_is_noop() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();
        let idx = table.claim_slot(1).unwrap();
        table.add_usage(idx, 0, 1000);

        // Out-of-bounds device_idx
        table.add_usage(idx, MAX_DEVICES, 500);
        table.add_usage(idx, usize::MAX, 500);
        assert_eq!(table.read_slot_usage(idx)[0], 1000);

        // Out-of-bounds slot_idx (should not panic)
        table.add_usage(MAX_PROC_SLOTS, 0, 1000);
        table.add_usage(usize::MAX, 0, 1000);
    }

    #[test]
    fn sub_usage_out_of_bounds_is_noop() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();
        let idx = table.claim_slot(1).unwrap();
        table.add_usage(idx, 0, 1000);

        // Out-of-bounds device_idx
        table.sub_usage(idx, MAX_DEVICES, 500);
        assert_eq!(table.read_slot_usage(idx)[0], 1000);

        // Out-of-bounds slot_idx (should not panic)
        table.sub_usage(MAX_PROC_SLOTS, 0, 500);
    }

    #[test]
    fn add_sub_usage_last_device_boundary() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();
        let idx = table.claim_slot(1).unwrap();

        // MAX_DEVICES - 1 is the last valid device index
        table.add_usage(idx, MAX_DEVICES - 1, 4200);
        assert_eq!(table.read_slot_usage(idx)[MAX_DEVICES - 1], 4200);

        table.sub_usage(idx, MAX_DEVICES - 1, 1200);
        assert_eq!(table.read_slot_usage(idx)[MAX_DEVICES - 1], 3000);

        // All other devices untouched
        for d in 0..MAX_DEVICES - 1 {
            assert_eq!(table.read_slot_usage(idx)[d], 0, "device {d} must be 0");
        }
    }

    #[test]
    fn read_slot_usage_out_of_bounds_returns_zeroes() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();

        // OOB slot_idx should return all zeroes, not panic
        let usage = table.read_slot_usage(MAX_PROC_SLOTS);
        assert_eq!(usage, [0u64; MAX_DEVICES]);

        let usage = table.read_slot_usage(usize::MAX);
        assert_eq!(usage, [0u64; MAX_DEVICES]);
    }

    #[test]
    fn consecutive_reaps_are_idempotent() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();

        let idx = table.claim_slot(4_000_000).unwrap();
        table.add_usage(idx, 0, 999);

        // First reap finds it.
        let dead1 = table.find_dead_slots();
        assert_eq!(dead1.len(), 1);

        // CAS-based reap succeeds.
        let usage = table.try_claim_and_release(idx, 4_000_000);
        assert!(usage.is_some());

        // Second find_dead_slots sees nothing (slot was released).
        let dead2 = table.find_dead_slots();
        assert!(dead2.is_empty(), "second reap should find nothing");
    }

    #[test]
    fn find_dead_slots_empty_when_no_dead() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();

        let pid = std::process::id();
        let idx = table.claim_slot(pid).unwrap();
        table.add_usage(idx, 0, 500);

        let dead = table.find_dead_slots();
        assert!(dead.is_empty());
        // Our usage is untouched.
        assert_eq!(table.read_slot_usage(idx)[0], 500);
    }

    #[test]
    fn zero_and_release_clears_all_devices() {
        let table = ProcSlotTable::new_zeroed();
        table.initialize();

        let idx = table.claim_slot(1).unwrap();
        table.add_usage(idx, 0, 100);
        table.add_usage(idx, 1, 200);
        table.add_usage(idx, 2, 300);

        table.zero_and_release(idx);

        assert_eq!(table.slots[idx].pid.load(Ordering::Relaxed), 0);
        let usage = table.read_slot_usage(idx);
        assert_eq!(usage[0], 0);
        assert_eq!(usage[1], 0);
        assert_eq!(usage[2], 0);
    }

    // -- ProcSlotHandle integration tests --

    #[test]
    fn handle_create_and_claim() {
        let tmp = TempDir::new().unwrap();
        let handle =
            ProcSlotHandle::create_and_claim(tmp.path()).expect("create_and_claim should succeed");
        assert!(handle.slot_idx().is_some());
        assert!(handle.table().is_initialized());
    }

    #[test]
    fn handle_add_sub_usage() {
        let tmp = TempDir::new().unwrap();
        let handle = ProcSlotHandle::create_and_claim(tmp.path()).unwrap();
        let idx = handle.slot_idx().unwrap();

        handle.add_usage(0, 1000);
        handle.add_usage(0, 500);
        handle.sub_usage(0, 300);

        let usage = handle.table().read_slot_usage(idx);
        assert_eq!(usage[0], 1200);
    }

    #[test]
    fn handle_drain_our_slot() {
        let tmp = TempDir::new().unwrap();
        let handle = ProcSlotHandle::create_and_claim(tmp.path()).unwrap();
        let idx = handle.slot_idx().unwrap();

        handle.add_usage(0, 100);
        handle.add_usage(1, 200);

        let usage = handle.drain_our_slot().expect("drain should return usage");
        assert_eq!(usage[0], 100);
        assert_eq!(usage[1], 200);

        // Slot should be released now.
        assert_eq!(
            handle.table().slots[idx].pid.load(Ordering::Relaxed),
            0,
            "slot should be released after drain"
        );

        // slot_idx should be invalidated.
        assert!(handle.slot_idx().is_none(), "slot_idx should be None after drain");

        // Subsequent add_usage/sub_usage are no-ops.
        handle.add_usage(0, 9999);
        assert_eq!(handle.table().read_slot_usage(idx)[0], 0, "add_usage after drain should be noop");
    }

    #[test]
    fn handle_reap_dead() {
        let tmp = TempDir::new().unwrap();
        let handle = ProcSlotHandle::create_and_claim(tmp.path()).unwrap();

        // Manually inject a dead process into the same SHM table
        let dead_pid: u32 = 4_000_000;
        let dead_idx = handle.table().claim_slot(dead_pid).unwrap();
        handle.table().add_usage(dead_idx, 0, 5000);
        handle.table().add_usage(dead_idx, 1, 3000);

        // reap_dead through the handle API
        let reaped = handle.reap_dead();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].0, dead_pid);
        assert_eq!(reaped[0].1[0], 5000);
        assert_eq!(reaped[0].1[1], 3000);

        // Dead slot is released, our slot is untouched
        assert_eq!(handle.table().slots[dead_idx].pid.load(Ordering::Relaxed), 0);
        assert!(handle.slot_idx().is_some(), "our slot must still be valid");

        // Second reap finds nothing
        let reaped2 = handle.reap_dead();
        assert!(reaped2.is_empty());
    }
}
