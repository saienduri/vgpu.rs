use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use erl::KernelLimiter;
use once_cell::sync::OnceCell;
use utils::shared_memory::{erl_adapter::ErlSharedMemoryAdapter, handle::SharedMemoryHandle};

use crate::hiplib::{self, HipDevice, HipError};

/// Isolation mode that activates memory enforcement hooks.
pub(crate) const ISOLATION_SOFT: &str = "soft";

#[derive(thiserror::Error, Debug)]
pub(crate) enum Error {
    #[error("HIP error: {0}")]
    Hip(HipError),

    #[error("Shared memory access failed: {0}")]
    SharedMemory(#[from] anyhow::Error),

    #[error("Device not configured: {0}")]
    DeviceNotConfigured(String),

    #[error("Allocation exceeds limit on device {device_idx}: used ({used}) + request ({request}) > limit ({limit})")]
    OverLimit {
        used: u64,
        request: u64,
        limit: u64,
        device_idx: usize,
    },

    #[error("Limiter not initialized")]
    LimiterNotInitialized,
}

pub(crate) struct Limiter {
    shared_memory_handle: OnceCell<Arc<SharedMemoryHandle>>,
    erl_kernel_limiter: OnceCell<KernelLimiter<ErlSharedMemoryAdapter<Arc<SharedMemoryHandle>>>>,
    /// Cache: HIP device ordinal -> (raw_device_index, device_uuid)
    hip_device_mapping: DashMap<HipDevice, (usize, String)>,
    /// Configured devices: (raw_device_index, device_uuid) sorted/deduped from config
    gpu_idx_uuids: Vec<(usize, String)>,
    isolation: Option<String>,
    /// Tracks pointer address -> (device_index, allocation_size) for free hooks.
    /// Process-local: pointer addresses are virtual and only meaningful within this process.
    /// The actual pod_memory_used counter lives in SHM (shared across processes).
    allocation_tracker: DashMap<usize, (usize, u64)>,
    /// When true, heartbeat warnings are suppressed (no hypervisor to heartbeat).
    standalone: bool,
    /// Monotonic allocation counter for periodic reconciliation diagnostics.
    alloc_count: AtomicU64,
}

impl std::fmt::Debug for Limiter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Limiter").finish()
    }
}

/// Normalize an AMD GPU UUID to a PCI BDF string for matching.
/// AMD GPU UUIDs are PCI BDF-based: "AMD-GPU-0000:03:00.0"
/// The Go hypervisor lowercases UUIDs, so we normalize to lowercase
/// and strip the "amd-gpu-" prefix.
pub(crate) fn normalize_uuid_to_bdf(uuid: &str) -> String {
    let lowered = uuid.to_lowercase();
    lowered
        .strip_prefix("amd-gpu-")
        .unwrap_or(&lowered)
        .to_string()
}

/// Match config GPU UUIDs against enumerated HIP devices by PCI bus ID.
/// Returns sorted (device_index, uuid) pairs for matched devices.
///
/// `gpu_uuids`: UUIDs from the hypervisor config (may have "AMD-GPU-" prefix, mixed case)
/// `enumerated_devices`: (device_index, pci_bus_id) pairs from HIP device enumeration
pub(crate) fn resolve_device_indices(
    gpu_uuids: &[String],
    enumerated_devices: &[(i32, String)],
) -> Vec<(usize, String)> {
    let mut resolved = Vec::new();
    for uuid in gpu_uuids {
        let target_bdf = normalize_uuid_to_bdf(uuid);
        if let Some((device_index, _)) = enumerated_devices
            .iter()
            .find(|(_, pci_bus_id)| pci_bus_id.to_lowercase() == target_bdf)
        {
            resolved.push((*device_index as usize, uuid.clone()));
        } else {
            tracing::warn!(uuid = uuid.as_str(), "No HIP device found matching UUID, skipping");
        }
    }
    resolved.sort_by_key(|(idx, _)| *idx);
    resolved.dedup_by_key(|(idx, _)| *idx);
    resolved
}

impl Limiter {
    pub(crate) fn new(
        mut gpu_uuids: Vec<String>,
        isolation: Option<String>,
        standalone: bool,
    ) -> Result<Self, Error> {
        gpu_uuids.sort();
        gpu_uuids.dedup();

        let hip = hiplib::hiplib();
        let device_count = hip.get_device_count().map_err(Error::Hip)?;

        let mut enumerated_devices = Vec::new();
        for device_index in 0..device_count {
            let pci_bus_id = hip.get_pci_bus_id(device_index).map_err(Error::Hip)?;
            enumerated_devices.push((device_index, pci_bus_id));
        }

        let gpu_idx_uuids = resolve_device_indices(&gpu_uuids, &enumerated_devices);

        tracing::info!(
            "Limiter initialized with GPU UUIDs and indices: {:?}",
            gpu_idx_uuids
        );

        Ok(Self {
            shared_memory_handle: OnceCell::new(),
            erl_kernel_limiter: OnceCell::new(),
            hip_device_mapping: DashMap::new(),
            gpu_idx_uuids,
            isolation,
            allocation_tracker: DashMap::new(),
            standalone,
            alloc_count: AtomicU64::new(0),
        })
    }

    /// Eagerly set the SHM handle (used by standalone mode which creates its own SHM).
    pub(crate) fn set_shared_memory_handle(&self, handle: SharedMemoryHandle) -> Result<(), Error> {
        self.shared_memory_handle
            .set(Arc::new(handle))
            .map_err(|_| Error::SharedMemory(anyhow::anyhow!("SHM handle already set")))
    }

    fn get_or_init_shared_memory(&self) -> Result<&SharedMemoryHandle, Error> {
        Ok(self.get_or_init_shared_memory_arc()?.as_ref())
    }

    fn get_or_init_shared_memory_arc(&self) -> Result<&Arc<SharedMemoryHandle>, Error> {
        self.shared_memory_handle.get_or_try_init(|| {
            if let Some(shm_path) = crate::mock_shm_path() {
                Ok(Arc::new(SharedMemoryHandle::mock(
                    shm_path,
                    self.gpu_idx_uuids.clone(),
                )))
            } else {
                SharedMemoryHandle::open(shm_path())
                    .map(Arc::new)
                    .map_err(Error::SharedMemory)
            }
        })
    }

    /// Used by ERL compute throttling (not yet wired for hip-limiter).
    #[allow(dead_code)]
    fn get_or_init_kernel_limiter(
        &self,
    ) -> Result<&KernelLimiter<ErlSharedMemoryAdapter<Arc<SharedMemoryHandle>>>, Error> {
        self.erl_kernel_limiter.get_or_try_init(|| {
            let handle = Arc::clone(self.get_or_init_shared_memory_arc()?);
            let erl_adapter = ErlSharedMemoryAdapter::new(handle);
            Ok(KernelLimiter::new(erl_adapter))
        })
    }

    pub(crate) fn device_index_by_hip_device(
        &self,
        hip_device: HipDevice,
    ) -> Result<usize, Error> {
        if let Some(entry) = self.hip_device_mapping.get(&hip_device) {
            return Ok(entry.0);
        }

        // Look up via PCI bus ID
        let hip = hiplib::hiplib();
        let pci_bus_id = hip.get_pci_bus_id(hip_device).map_err(Error::Hip)?;
        let pci_bus_id_lower = pci_bus_id.to_lowercase();

        for (idx, uuid) in &self.gpu_idx_uuids {
            let target_bdf = normalize_uuid_to_bdf(uuid);
            if pci_bus_id_lower == target_bdf {
                self.hip_device_mapping
                    .insert(hip_device, (*idx, uuid.clone()));
                return Ok(*idx);
            }
        }

        Err(Error::DeviceNotConfigured(format!("HIP device {hip_device}")))
    }

    pub(crate) fn get_pod_memory_usage(
        &self,
        raw_device_index: usize,
    ) -> Result<(u64, u64), Error> {
        let handle = self.get_or_init_shared_memory()?;
        let state = handle.get_state();

        if !self.standalone && !state.is_healthy(Duration::from_secs(2)) {
            tracing::warn!(
                device_idx = raw_device_index,
                last_heartbeat = state.get_last_heartbeat(),
                "Stale heartbeat detected, continuing with enforcement"
            );
        }

        if let Some((used, limit)) = state.with_device_v2_or(
            raw_device_index,
            |device| {
                (
                    device.device_info.get_pod_memory_used(),
                    device.device_info.get_mem_limit(),
                )
            },
        ) {
            Ok((used, limit))
        } else {
            Err(Error::DeviceNotConfigured(format!("SHM device {raw_device_index}")))
        }
    }

    /// Atomically reserve memory by incrementing pod_memory_used BEFORE calling the
    /// native allocator. Returns Ok(previous_used) if the reservation fits within the
    /// limit, or Err if it would exceed the limit (and rolls back the increment).
    ///
    /// This eliminates the TOCTOU race in the old check-then-allocate pattern: the
    /// atomic fetch_add IS the reservation, so concurrent threads cannot both pass
    /// the limit check with stale values.
    ///
    /// Under high contention with tight limits, multiple threads may each fetch_add
    /// past the limit simultaneously and all roll back, causing under-utilization
    /// (fewer successes than slots available). This is safe — conservative direction.
    pub(crate) fn try_reserve(
        &self,
        device_idx: usize,
        size: u64,
    ) -> Result<u64, Error> {
        if size == 0 {
            return Ok(0);
        }
        // Guard against u64 overflow: fetch_add wraps modularly, so a huge size
        // would corrupt pod_memory_used transiently until rollback. Any realistic
        // GPU allocation is well under this threshold.
        const MAX_ALLOC_SIZE: u64 = u64::MAX / 2;
        if size > MAX_ALLOC_SIZE {
            return Err(Error::OverLimit {
                used: 0,
                request: size,
                limit: 0,
                device_idx,
            });
        }
        let handle = self.get_or_init_shared_memory()?;
        let state = handle.get_state();

        if !self.standalone && !state.is_healthy(Duration::from_secs(2)) {
            tracing::warn!(
                device_idx = device_idx,
                last_heartbeat = state.get_last_heartbeat(),
                "Stale heartbeat detected, continuing with enforcement"
            );
        }

        // NOTE: Between this fetch_add and the potential rollback fetch_sub below,
        // pod_memory_used holds a transiently elevated value (actual_used + size).
        // A concurrent hipMemGetInfo reader may see slightly less free memory than
        // reality during this nanosecond-scale window. This is conservative (never
        // over-reports free memory) and acceptable for lock-free atomics.
        //
        // Both fetch_add and get_mem_limit are read in a single with_device call
        // to avoid a race where the device becomes unavailable between calls.
        let reserve_result = state.with_device_v2_or(
            device_idx,
            |device| {
                let previous_used = device.device_info.pod_memory_used.fetch_add(size, Ordering::AcqRel);
                let mem_limit = device.device_info.get_mem_limit();
                (previous_used, mem_limit)
            },
        );

        let Some((previous_used, mem_limit)) = reserve_result else {
            return Err(Error::DeviceNotConfigured(format!("SHM device {device_idx}")));
        };

        let new_used = previous_used.saturating_add(size);

        if new_used > mem_limit {
            // Over limit — roll back the reservation
            state.with_device_v2_or(
                device_idx,
                |device| device.device_info.saturating_fetch_sub_pod_memory_used(size),
            );
            return Err(Error::OverLimit {
                used: previous_used,
                request: size,
                limit: mem_limit,
                device_idx,
            });
        }

        Ok(previous_used)
    }

    /// Roll back a reservation when the native allocator fails after try_reserve succeeded.
    pub(crate) fn rollback_reservation(
        &self,
        device_idx: usize,
        size: u64,
    ) {
        if size == 0 {
            return;
        }
        let handle = match self.get_or_init_shared_memory() {
            Ok(handle) => handle,
            Err(error) => {
                tracing::warn!("Cannot rollback reservation, SHM unavailable: {error}");
                return;
            }
        };
        let state = handle.get_state();
        state.with_device_v2_or(
            device_idx,
            |device| device.device_info.saturating_fetch_sub_pod_memory_used(size),
        );
    }

    /// Record a successful allocation in the pointer tracker (after try_reserve + native alloc).
    /// The SHM pod_memory_used was already incremented by try_reserve.
    pub(crate) fn record_allocation(
        &self,
        device_idx: usize,
        ptr: usize,
        size: u64,
    ) {
        if size == 0 {
            return;
        }
        self.allocation_tracker.insert(ptr, (device_idx, size));

        // Periodic reconciliation: compare our counter with real VRAM usage
        let count = self.alloc_count.fetch_add(1, Ordering::Relaxed);
        if count % 100 == 0 {
            self.log_reconciliation(device_idx, count);
        }
    }

    /// Compare pod_memory_used (our counter) with real VRAM from the original hipMemGetInfo.
    /// Logs when they diverge, helping diagnose accounting drift.
    ///
    /// Emits three key metrics:
    /// - `shm_vs_real_mib`: SHM counter minus real VRAM (positive = SHM over-reports)
    /// - `tracker_vs_shm_mib`: DashMap sum minus SHM (negative = stale residual from other processes)
    /// - `stale_mib`: at alloc_count=0, shows how much SHM was already non-zero before this process
    fn log_reconciliation(&self, device_idx: usize, alloc_count: u64) {
        use crate::detour::mem::FN_HIP_MEM_GET_INFO;

        let pid = std::process::id();

        // Guard: hook may not be initialized yet during early startup
        let Some(hip_mem_get_info) = FN_HIP_MEM_GET_INFO.get() else {
            return;
        };

        let (our_used, mem_limit) = self
            .get_pod_memory_usage(device_idx)
            .map(|(used, limit)| (used, limit))
            .unwrap_or((0, 0));

        let mut real_free: usize = 0;
        let mut real_total: usize = 0;
        let result = unsafe { hip_mem_get_info(&mut real_free, &mut real_total) };
        if result != 0 {
            return; // native call failed, skip
        }
        let real_used = real_total.saturating_sub(real_free) as u64;
        let tracked_count = self.allocation_tracker.len();
        // Sum of all sizes in our process-local DashMap for this device
        let tracked_bytes: u64 = self
            .allocation_tracker
            .iter()
            .filter(|entry| entry.value().0 == device_idx)
            .map(|entry| entry.value().1)
            .sum();

        let shm_vs_real = our_used as i128 - real_used as i128;
        let shm_vs_real_mib = shm_vs_real / (1024 * 1024);
        let tracker_vs_shm = tracked_bytes as i128 - our_used as i128;
        let tracker_vs_shm_mib = tracker_vs_shm / (1024 * 1024);

        // First alloc: detect stale SHM from prior processes that didn't drain
        if alloc_count == 0 {
            let stale = our_used.saturating_sub(tracked_bytes);
            let stale_mib = stale / (1024 * 1024);
            if stale_mib > 0 {
                tracing::warn!(
                    pid,
                    device_idx,
                    our_shm = our_used,
                    real_vram = real_used,
                    tracked_bytes,
                    mem_limit,
                    stale_mib,
                    "STALE SHM: counter is {stale_mib} MiB above this process's tracked total \
                     on first alloc — prior process(es) likely exited without drain"
                );
            } else {
                tracing::info!(
                    pid,
                    device_idx,
                    our_shm = our_used,
                    real_vram = real_used,
                    tracked_bytes,
                    mem_limit,
                    "initial SHM state on first alloc"
                );
            }
            return;
        }

        if shm_vs_real_mib.abs() > 100 || tracker_vs_shm_mib.abs() > 100 {
            tracing::warn!(
                pid,
                alloc_count,
                device_idx,
                our_shm = our_used,
                real_vram = real_used,
                tracked_bytes,
                tracked_count,
                shm_vs_real_mib,
                tracker_vs_shm_mib,
                "RECONCILIATION DRIFT: SHM vs real VRAM = {shm_vs_real_mib} MiB, \
                 tracker vs SHM = {tracker_vs_shm_mib} MiB"
            );
        } else {
            tracing::info!(
                pid,
                alloc_count,
                device_idx,
                our_shm = our_used,
                real_vram = real_used,
                tracked_bytes,
                tracked_count,
                shm_vs_real_mib,
                tracker_vs_shm_mib,
                "reconciliation check"
            );
        }
    }

    /// Record a free: look up the pointer's size, decrement SHM pod_memory_used.
    /// Returns true if the pointer was tracked (and decremented), false if unknown.
    pub(crate) fn record_free(&self, ptr: usize) -> bool {
        let Some((_, (device_idx, size))) = self.allocation_tracker.remove(&ptr) else {
            return false;
        };
        let handle = match self.get_or_init_shared_memory() {
            Ok(handle) => handle,
            Err(error) => {
                // KNOWN LIMITATION: The pointer was removed from allocation_tracker but
                // pod_memory_used was not decremented. This causes a permanent accounting
                // leak of `size` bytes in SHM. However, SHM being unavailable indicates
                // the hypervisor is already in a degraded state, so this is acceptable.
                // Re-inserting the entry would cause a double-free on retry.
                tracing::warn!(size = size, device_idx = device_idx, "Cannot record free, SHM unavailable: {error}");
                return true;
            }
        };
        let state = handle.get_state();
        state.with_device_v2_or(
            device_idx,
            |device| device.device_info.saturating_fetch_sub_pod_memory_used(size),
        );
        true
    }

    /// Drain all tracked allocations and decrement SHM counters.
    /// Called at process exit (via `libc::atexit`) to prevent stale
    /// `pod_memory_used` accumulation across sequential processes sharing
    /// the same SHM segment.
    ///
    /// Aggregates per-device totals from the DashMap, then does one bulk
    /// `saturating_fetch_sub` per device (avoids N atomic ops for N allocations).
    ///
    /// Uses `eprintln` instead of `tracing` because Rust's TLS destructors
    /// may have already run by the time `atexit` fires, making the tracing
    /// subscriber inaccessible.
    pub(crate) fn drain_allocations(&self) {
        let pid = std::process::id();
        let handle = match self.shared_memory_handle.get() {
            Some(handle) => handle,
            None => return, // SHM was never initialized — nothing to drain
        };

        // Aggregate per-device totals
        let mut device_totals: std::collections::HashMap<usize, u64> =
            std::collections::HashMap::new();
        // Drain the tracker — removes all entries
        self.allocation_tracker.retain(|_, (device_idx, size)| {
            *device_totals.entry(*device_idx).or_default() += *size;
            false // remove every entry
        });

        if device_totals.is_empty() {
            return;
        }

        let state = handle.get_state();

        // Snapshot SHM before drain, drain, then snapshot after — shows the effect
        let mut drain_details: Vec<String> = Vec::new();
        for (device_idx, total_size) in &device_totals {
            let before = state
                .with_device_v2_or(*device_idx, |device| {
                    device.device_info.get_pod_memory_used()
                })
                .unwrap_or(0);
            state.with_device_v2_or(*device_idx, |device| {
                device
                    .device_info
                    .saturating_fetch_sub_pod_memory_used(*total_size)
            });
            let after = state
                .with_device_v2_or(*device_idx, |device| {
                    device.device_info.get_pod_memory_used()
                })
                .unwrap_or(0);
            drain_details.push(format!(
                "dev{device_idx}: drained {} MiB (shm: {} -> {} MiB)",
                total_size / (1024 * 1024),
                before / (1024 * 1024),
                after / (1024 * 1024),
            ));
        }

        let total_bytes: u64 = device_totals.values().sum();
        let alloc_count = self.alloc_count.load(Ordering::Relaxed);
        eprintln!(
            "[hip-limiter] drain (pid {pid}): {total_bytes} bytes ({} MiB), \
             {alloc_count} allocs tracked this process: [{}]",
            total_bytes / (1024 * 1024),
            drain_details.join(", ")
        );
    }

    pub(crate) fn isolation(&self) -> Option<&str> {
        self.isolation.as_deref()
    }

    /// Match a PCI BDF string (e.g., "0000:75:00.0") against configured devices.
    /// Used by amdsmi hooks to resolve opaque processor handles to SHM device indices.
    pub(crate) fn device_index_by_pci_bdf(&self, bdf: &str) -> Result<usize, Error> {
        let bdf_lower = bdf.to_lowercase();
        for (idx, uuid) in &self.gpu_idx_uuids {
            if normalize_uuid_to_bdf(uuid) == bdf_lower {
                return Ok(*idx);
            }
        }
        Err(Error::DeviceNotConfigured(format!("BDF {bdf}")))
    }

    pub(crate) fn all_devices_unlimited(&self) -> bool {
        let handle = match self.get_or_init_shared_memory() {
            Ok(handle) => handle,
            Err(error) => {
                tracing::warn!("Failed to access shared memory: {error}");
                return false;
            }
        };

        let state = handle.get_state();

        for (idx, _uuid) in &self.gpu_idx_uuids {
            let up_limit = state
                .with_device_v2_or(
                    *idx,
                    |device| device.device_info.get_up_limit(),
                )
                .unwrap_or(0);

            if up_limit < 100 {
                return false;
            }
        }

        true
    }
}

fn shm_path() -> PathBuf {
    PathBuf::from(crate::resolve_shm_path(false))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use utils::shared_memory::SharedDeviceInfo;

    trait SharedMemory {
        fn get_state(&self) -> &SharedDeviceInfo;
    }

    pub struct MockSharedMemory {
        state: Arc<SharedDeviceInfo>,
    }

    impl MockSharedMemory {
        pub fn new(total_cores: u32, up_limit: u32, mem_limit: u64) -> Self {
            let state = Arc::new(SharedDeviceInfo::new(total_cores, up_limit, mem_limit));
            Self { state }
        }

        pub fn set_pod_memory_used(&self, used: u64) {
            self.state.set_pod_memory_used(used);
        }
    }

    impl SharedMemory for MockSharedMemory {
        fn get_state(&self) -> &SharedDeviceInfo {
            &self.state
        }
    }

    /// Smoke test: verify get/set on SharedDeviceInfo and free = limit - used.
    #[test]
    fn test_memory_info_reporting() {
        let mock = MockSharedMemory::new(1000, 80, 1024 * 1024 * 1024);
        mock.set_pod_memory_used(512 * 1024 * 1024);

        let state = mock.get_state();
        let total = state.get_mem_limit();
        let used = state.get_pod_memory_used();
        let free = total.saturating_sub(used);

        assert_eq!(total, 1024 * 1024 * 1024);
        assert_eq!(used, 512 * 1024 * 1024);
        assert_eq!(free, 512 * 1024 * 1024);
    }

    /// Edge case: saturating_sub prevents underflow when used >= limit.
    #[test]
    fn test_memory_info_edge_cases() {
        let mock = MockSharedMemory::new(1000, 80, 1024);
        mock.set_pod_memory_used(1024);

        let state = mock.get_state();
        let total = state.get_mem_limit();
        let used = state.get_pod_memory_used();
        let free = total.saturating_sub(used);

        assert_eq!(free, 0);

        // When used exceeds total, saturating_sub prevents underflow
        mock.set_pod_memory_used(2048);
        let used = state.get_pod_memory_used();
        let free = total.saturating_sub(used);
        assert_eq!(free, 0);
    }

    // --- UUID normalization tests ---
    //
    // Real formats from production:
    //   HIP runtime returns:     "0000:75:00.0" (lowercase hex, domain:bus:device.function)
    //   Go hypervisor constructs: strings.ToLower("AMD-GPU-" + BDF) = "amd-gpu-0000:75:00.0"
    //   C provider uses:          snprintf("AMD-GPU-%04x:%02x:%02x.%x") = "AMD-GPU-0000:75:00.0"
    //
    // The normalizer must handle all three sources.

    #[test]
    fn test_normalize_uuid_strips_lowercase_prefix() {
        // Go hypervisor format (always lowercase)
        assert_eq!(
            super::normalize_uuid_to_bdf("amd-gpu-0000:75:00.0"),
            "0000:75:00.0"
        );
    }

    #[test]
    fn test_normalize_uuid_strips_uppercase_prefix() {
        // C provider format (uppercase prefix, lowercase hex)
        assert_eq!(
            super::normalize_uuid_to_bdf("AMD-GPU-0000:75:00.0"),
            "0000:75:00.0"
        );
    }

    #[test]
    fn test_normalize_uuid_bare_bdf() {
        // Raw PCI bus ID as HIP returns it
        assert_eq!(
            super::normalize_uuid_to_bdf("0000:75:00.0"),
            "0000:75:00.0"
        );
    }

    #[test]
    fn test_normalize_uuid_mixed_case_hex() {
        // Hypothetical: uppercase hex digits in bus ID
        assert_eq!(
            super::normalize_uuid_to_bdf("AMD-GPU-0000:F5:00.0"),
            "0000:f5:00.0"
        );
    }

    // --- Device resolution tests ---
    //
    // Uses real MI325X PCI bus IDs from production 8-GPU node:
    //   Device 0: 0000:75:00.0    Device 4: 0000:f5:00.0
    //   Device 1: 0000:05:00.0    Device 5: 0000:85:00.0
    //   Device 2: 0000:65:00.0    Device 6: 0000:e5:00.0
    //   Device 3: 0000:15:00.0    Device 7: 0000:95:00.0

    fn devices(pairs: &[(i32, &str)]) -> Vec<(i32, String)> {
        pairs.iter().map(|(i, s)| (*i, s.to_string())).collect()
    }

    fn uuids(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    /// Real MI325X 8-GPU enumeration
    fn mi325x_devices() -> Vec<(i32, String)> {
        devices(&[
            (0, "0000:75:00.0"),
            (1, "0000:05:00.0"),
            (2, "0000:65:00.0"),
            (3, "0000:15:00.0"),
            (4, "0000:f5:00.0"),
            (5, "0000:85:00.0"),
            (6, "0000:e5:00.0"),
            (7, "0000:95:00.0"),
        ])
    }

    #[test]
    fn test_resolve_picks_correct_gpus_on_mi325x() {
        // Pod allocated GPUs 0 and 4 (PCI slots 75 and f5)
        let config = uuids(&["amd-gpu-0000:75:00.0", "amd-gpu-0000:f5:00.0"]);
        let result = super::resolve_device_indices(&config, &mi325x_devices());

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0); // device 0 = 0000:75:00.0
        assert_eq!(result[1].0, 4); // device 4 = 0000:f5:00.0
    }

    #[test]
    fn test_resolve_single_gpu_allocation() {
        // Pod allocated only GPU 3 (PCI slot 15)
        let config = uuids(&["amd-gpu-0000:15:00.0"]);
        let result = super::resolve_device_indices(&config, &mi325x_devices());

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 3);
    }

    #[test]
    fn test_resolve_all_8_gpus() {
        // Pod allocated all 8 GPUs
        let config = uuids(&[
            "amd-gpu-0000:75:00.0",
            "amd-gpu-0000:05:00.0",
            "amd-gpu-0000:65:00.0",
            "amd-gpu-0000:15:00.0",
            "amd-gpu-0000:f5:00.0",
            "amd-gpu-0000:85:00.0",
            "amd-gpu-0000:e5:00.0",
            "amd-gpu-0000:95:00.0",
        ]);
        let result = super::resolve_device_indices(&config, &mi325x_devices());

        assert_eq!(result.len(), 8);
        // Should be sorted by device index
        let indices: Vec<usize> = result.iter().map(|(idx, _)| *idx).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_resolve_c_provider_format() {
        // C provider uses uppercase prefix: "AMD-GPU-0000:75:00.0"
        let config = uuids(&["AMD-GPU-0000:85:00.0"]);
        let result = super::resolve_device_indices(&config, &mi325x_devices());

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 5); // device 5 = 0000:85:00.0
    }

    #[test]
    fn test_resolve_missing_uuid_skipped() {
        // One real device, one that doesn't exist on this node
        let config = uuids(&["amd-gpu-0000:75:00.0", "amd-gpu-0000:aa:00.0"]);
        let result = super::resolve_device_indices(&config, &mi325x_devices());

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
    }

    #[test]
    fn test_resolve_empty_config() {
        let result = super::resolve_device_indices(&uuids(&[]), &mi325x_devices());
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_empty_enumerated() {
        let config = uuids(&["amd-gpu-0000:75:00.0"]);
        let result = super::resolve_device_indices(&config, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_sorted_regardless_of_config_order() {
        // Config lists devices in reverse order; result should still be sorted by index
        let config = uuids(&[
            "amd-gpu-0000:95:00.0", // device 7
            "amd-gpu-0000:05:00.0", // device 1
            "amd-gpu-0000:e5:00.0", // device 6
        ]);
        let result = super::resolve_device_indices(&config, &mi325x_devices());

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, 1); // device 1
        assert_eq!(result[1].0, 6); // device 6
        assert_eq!(result[2].0, 7); // device 7
    }

    #[test]
    fn test_resolve_device_indices_bdf_match() {
        let result = super::resolve_device_indices(
            &uuids(&["amd-gpu-0000:75:00.0", "amd-gpu-0000:f5:00.0"]),
            &mi325x_devices(),
        );
        assert_eq!(result.len(), 2);

        let bdf = "0000:75:00.0";
        let bdf_lower = bdf.to_lowercase();
        let found = result.iter().find(|(_, uuid)| {
            super::normalize_uuid_to_bdf(uuid) == bdf_lower
        });
        assert!(found.is_some());
        assert_eq!(found.unwrap().0, 0);
    }

    #[test]
    fn test_resolve_device_indices_bdf_no_match() {
        let result = super::resolve_device_indices(
            &uuids(&["amd-gpu-0000:75:00.0"]),
            &mi325x_devices(),
        );
        let bdf = "0000:aa:00.0";
        let bdf_lower = bdf.to_lowercase();
        let found = result.iter().find(|(_, uuid)| {
            super::normalize_uuid_to_bdf(uuid) == bdf_lower
        });
        assert!(found.is_none());
    }

    // --- BDF formatting tests (amdsmi bitfield → PCI bus ID string) ---
    //
    // amdsmi_bdf_t layout: function(3) | device(5) | bus(8) | domain(48)
    // Verified against MI325X empirical data.

    #[test]
    fn test_format_amdsmi_bdf_mi325x_device() {
        // MI325X GPU at 0000:75:00.0 → raw = 0x7500
        use crate::detour::smi::format_amdsmi_bdf;
        assert_eq!(format_amdsmi_bdf(0x0000000000007500), "0000:75:00.0");
    }

    #[test]
    fn test_format_amdsmi_bdf_high_bus() {
        // MI325X GPU at 0000:f5:00.0 → raw = 0xf500
        use crate::detour::smi::format_amdsmi_bdf;
        assert_eq!(format_amdsmi_bdf(0x000000000000f500), "0000:f5:00.0");
    }

    #[test]
    fn test_format_amdsmi_bdf_with_function() {
        // Device with function number 3: 0000:05:01.3
        use crate::detour::smi::format_amdsmi_bdf;
        let raw: u64 = 3 | (1 << 3) | (0x05 << 8);
        assert_eq!(format_amdsmi_bdf(raw), "0000:05:01.3");
    }

    #[test]
    fn test_format_amdsmi_bdf_nonzero_domain() {
        // Multi-domain system: 0001:75:00.0
        use crate::detour::smi::format_amdsmi_bdf;
        let raw: u64 = (0x75 << 8) | (1u64 << 16);
        assert_eq!(format_amdsmi_bdf(raw), "0001:75:00.0");
    }
}
