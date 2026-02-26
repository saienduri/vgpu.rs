use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use erl::KernelLimiter;
use once_cell::sync::OnceCell;
use utils::shared_memory::{erl_adapter::ErlSharedMemoryAdapter, handle::SharedMemoryHandle};

use crate::hiplib::{self, HipDevice, HipError};

#[derive(thiserror::Error, Debug)]
pub(crate) enum Error {
    #[error("HIP error: {0}")]
    Hip(HipError),

    #[error("Shared memory access failed: {0}")]
    SharedMemory(#[from] anyhow::Error),

    #[error("Device {0} not configured")]
    DeviceNotConfigured(usize),

    #[error("Device {device_idx} not healthy, last heartbeat {last_heartbeat}")]
    DeviceNotHealthy {
        device_idx: usize,
        last_heartbeat: u64,
    },

    #[error("Limiter not initialized")]
    LimiterNotInitialized,
}

pub(crate) struct Limiter {
    shared_memory_handle: OnceCell<Arc<SharedMemoryHandle>>,
    erl_kernel_limiter: OnceCell<KernelLimiter<ErlSharedMemoryAdapter<Arc<SharedMemoryHandle>>>>,
    /// HIP device -> (raw_device_index, device_uuid)
    hip_device_mapping: DashMap<HipDevice, (usize, String)>,
    gpu_idx_uuids: Vec<(usize, String)>,
    isolation: Option<String>,
}

impl std::fmt::Debug for Limiter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Limiter").finish()
    }
}

impl Limiter {
    pub(crate) fn new(
        mut gpu_uuids: Vec<String>,
        isolation: Option<String>,
    ) -> Result<Self, Error> {
        gpu_uuids.sort();
        gpu_uuids.dedup();

        let hip = hiplib::hiplib();
        let device_count = hip.get_device_count().map_err(Error::Hip)?;

        let mut gpu_idx_uuids = Vec::new();
        for uuid in &gpu_uuids {
            // AMD GPU UUIDs are PCI BDF-based: "AMD-GPU-0000:03:00.0"
            // The Go hypervisor lowercases UUIDs, so handle both cases.
            let lowered = uuid.to_lowercase();
            let target_bdf = lowered
                .strip_prefix("amd-gpu-")
                .unwrap_or(&lowered);

            let mut found = false;
            for device_index in 0..device_count {
                let pci_bus_id = hip
                    .get_pci_bus_id(device_index)
                    .map_err(Error::Hip)?
                    .to_lowercase();

                if pci_bus_id == target_bdf {
                    gpu_idx_uuids.push((device_index as usize, uuid.clone()));
                    found = true;
                    break;
                }
            }

            if !found {
                tracing::warn!(
                    uuid = uuid.as_str(),
                    "No HIP device found matching UUID, skipping"
                );
            }
        }

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
        })
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
            let lowered = uuid.to_lowercase();
            let target_bdf = lowered
                .strip_prefix("amd-gpu-")
                .unwrap_or(&lowered);
            if pci_bus_id_lower == target_bdf {
                self.hip_device_mapping
                    .insert(hip_device, (*idx, uuid.clone()));
                return Ok(*idx);
            }
        }

        Err(Error::DeviceNotConfigured(hip_device as usize))
    }

    pub(crate) fn get_pod_memory_usage(
        &self,
        raw_device_index: usize,
    ) -> Result<(u64, u64), Error> {
        let handle = self.get_or_init_shared_memory()?;
        let state = handle.get_state();

        if !state.is_healthy(Duration::from_secs(2)) {
            let last_heartbeat = state.get_last_heartbeat();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            tracing::warn!(
                now = now,
                "{}",
                Error::DeviceNotHealthy {
                    device_idx: raw_device_index,
                    last_heartbeat,
                }
            );
        }

        if let Some((used, limit)) = state.with_device(
            raw_device_index,
            |device| {
                (
                    device.device_info.get_pod_memory_used(),
                    device.device_info.get_mem_limit(),
                )
            },
            |device| {
                (
                    device.device_info.get_pod_memory_used(),
                    device.device_info.get_mem_limit(),
                )
            },
        ) {
            Ok((used, limit))
        } else {
            Err(Error::DeviceNotConfigured(raw_device_index))
        }
    }

    pub(crate) fn isolation(&self) -> Option<&str> {
        self.isolation.as_deref()
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
                .with_device(
                    *idx,
                    |device| device.device_info.get_up_limit(),
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
    PathBuf::from(
        &std::env::var("SHM_PATH").unwrap_or_else(|_| "/run/tensor-fusion/shm".to_string()),
    )
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
}
