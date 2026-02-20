use std::cell::Cell;
use std::collections::HashSet;
use std::env;
use std::ffi::c_char;
use std::ffi::c_void;
use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;
use std::sync::OnceLock;

use ctor::ctor;
use limiter::Limiter;
use tf_macro::hook_fn;
use utils::hooks::HookManager;
use utils::logging;
use utils::replace_symbol;

mod config;
mod detour;
mod hiplib;
mod limiter;

static GLOBAL_LIMITER: OnceLock<Limiter> = OnceLock::new();
static GLOBAL_LIMITER_ERROR: OnceLock<String> = OnceLock::new();
static HOOKS_INITIALIZED: AtomicBool = AtomicBool::new(false);
static LIMITER_ERROR_REPORTED: AtomicBool = AtomicBool::new(false);

#[ctor]
unsafe fn entry_point() {
    logging::init();

    let enable_hip_hooks = env::var("ENABLE_HIP_HOOKS")
        .map(|value| value != "false")
        .unwrap_or(true);

    tracing::info!("enable_hip_hooks: {enable_hip_hooks}");

    if !enable_hip_hooks {
        HOOKS_INITIALIZED.store(true, Ordering::Release);
    }

    init_hooks();
}

fn should_skip_hooks_on_no_limit() -> bool {
    static SKIP_HOOKS_ON_NO_LIMIT: OnceLock<bool> = OnceLock::new();
    *SKIP_HOOKS_ON_NO_LIMIT.get_or_init(|| {
        env::var("TF_SKIP_HOOKS_IF_NO_LIMIT")
            .map(|value| value == "true" || value == "1")
            .unwrap_or(false)
    })
}

fn record_limiter_error(message: impl Into<String>) {
    let message = message.into();
    tracing::error!("{message}");
    if GLOBAL_LIMITER_ERROR.set(message).is_err() {
        tracing::debug!("Limiter error already recorded");
    }
}

pub(crate) fn report_limiter_not_initialized() {
    if LIMITER_ERROR_REPORTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
        .is_ok()
    {
        if let Some(reason) = GLOBAL_LIMITER_ERROR.get() {
            tracing::warn!("Limiter not initialized; last error: {reason}");
        } else {
            tracing::warn!("Limiter not initialized; init has not run");
        }
    }
}

pub(crate) fn mock_shm_path() -> Option<PathBuf> {
    env::var("TF_SHM_FILE")
        .map(PathBuf::from)
        .map(|mut path| {
            path.pop();
            path
        })
        .ok()
}

fn remap_visible_devices(allocated_devices: &[String]) -> Result<String, String> {
    if let Ok(last_remapped) = env::var("TF_REMAPPED") {
        if let Ok(current) = env::var("HIP_VISIBLE_DEVICES") {
            if current.trim() == last_remapped {
                return Ok(last_remapped);
            }
        } else {
            let result = allocated_devices.join(",");
            env::set_var("TF_REMAPPED", &result);
            return Ok(result);
        }
    }

    let original = env::var("HIP_VISIBLE_DEVICES").ok();
    let Some(original) = original else {
        let result = allocated_devices.join(",");
        env::set_var("TF_REMAPPED", &result);
        return Ok(result);
    };

    let trimmed = original.trim();
    if trimmed.is_empty() {
        let result = allocated_devices.join(",");
        env::set_var("TF_REMAPPED", &result);
        return Ok(result);
    }

    if trimmed.contains(',') {
        let mut remapped = Vec::new();
        for part in trimmed.split(',') {
            let virtual_id = part.trim().parse::<usize>().map_err(|_| {
                format!(
                    "Invalid device ID in HIP_VISIBLE_DEVICES: '{}'",
                    part.trim()
                )
            })?;
            if virtual_id >= allocated_devices.len() {
                return Err(format!(
                    "Virtual device ID {} out of range (only {} device(s) allocated)",
                    virtual_id,
                    allocated_devices.len()
                ));
            }
            remapped.push(allocated_devices[virtual_id].clone());
        }
        let result = remapped.join(",");
        env::set_var("TF_REMAPPED", &result);
        return Ok(result);
    }

    let virtual_id = trimmed
        .parse::<usize>()
        .map_err(|_| format!("Invalid device ID in HIP_VISIBLE_DEVICES: '{trimmed}'"))?;

    if virtual_id >= allocated_devices.len() {
        return Err(format!(
            "Virtual device ID {} out of range (only {} device(s) allocated)",
            virtual_id,
            allocated_devices.len()
        ));
    }

    let result = allocated_devices[virtual_id].clone();
    env::set_var("TF_REMAPPED", &result);
    Ok(result)
}

fn init_limiter() {
    static LIMITER_INITIALIZED: Once = Once::new();
    LIMITER_INITIALIZED.call_once(|| {
        let hip = match hiplib::init_hiplib() {
            Ok(hip) => hip,
            Err(error) => {
                record_limiter_error(format!("failed to initialize HIP library: {error}"));
                return;
            }
        };

        let config = if mock_shm_path().is_none() {
            let (hypervisor_ip, hypervisor_port) = match config::get_hypervisor_config() {
                Some((ip, port)) => (ip, port),
                None => {
                    record_limiter_error(
                        "HYPERVISOR_IP or HYPERVISOR_PORT not set; skipping limiter init",
                    );
                    return;
                }
            };

            match config::get_worker_config(&hypervisor_ip, &hypervisor_port) {
                Ok(config) => config,
                Err(error) => {
                    record_limiter_error(format!("failed to get device configs: {error}"));
                    return;
                }
            }
        } else {
            let uuids = match env::var("TF_VISIBLE_DEVICES") {
                Ok(visible_devices) => visible_devices
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>(),
                Err(_) => {
                    record_limiter_error(
                        "TF_VISIBLE_DEVICES not set in mock/test mode; skipping limiter init",
                    );
                    return;
                }
            };

            config::PodConfig {
                gpu_uuids: uuids,
                isolation: None,
            }
        };

        if !config.gpu_uuids.is_empty() {
            let device_count = match hip.get_device_count() {
                Ok(count) => count,
                Err(error) => {
                    record_limiter_error(format!("failed to get HIP device count: {error}"));
                    return;
                }
            };

            let lower_case_uuids: HashSet<_> = config
                .gpu_uuids
                .iter()
                .map(|uuid| {
                    uuid.strip_prefix("AMD-GPU-")
                        .unwrap_or(uuid)
                        .to_lowercase()
                })
                .collect();

            let mut device_indices = Vec::new();
            for device_index in 0..device_count {
                let pci_bus_id = match hip.get_pci_bus_id(device_index) {
                    Ok(id) => id.to_lowercase(),
                    Err(error) => {
                        record_limiter_error(format!(
                            "failed to get PCI bus ID for device {device_index}: {error}"
                        ));
                        return;
                    }
                };

                if lower_case_uuids.contains(&pci_bus_id) {
                    device_indices.push(device_index.to_string());
                }
            }

            if !device_indices.is_empty() {
                device_indices.sort_by_key(|id| id.parse::<u32>().unwrap_or(u32::MAX));

                let visible_devices = match remap_visible_devices(&device_indices) {
                    Ok(devices) => devices,
                    Err(error) => {
                        record_limiter_error(error);
                        return;
                    }
                };

                tracing::info!(
                    "Setting HIP_VISIBLE_DEVICES to {} (allocated devices: {})",
                    &visible_devices,
                    device_indices.join(",")
                );
                env::set_var("HIP_VISIBLE_DEVICES", &visible_devices);
            }
        }

        let limiter = match Limiter::new(config.gpu_uuids, config.isolation) {
            Ok(limiter) => limiter,
            Err(error) => {
                record_limiter_error(format!("failed to initialize limiter: {error}"));
                return;
            }
        };

        if GLOBAL_LIMITER.set(limiter).is_err() {
            record_limiter_error("GLOBAL_LIMITER already initialized");
        }
    });
}

fn try_install_hip_hooks() {
    if HOOKS_INITIALIZED.load(Ordering::Acquire) {
        return;
    }

    if !utils::hooks::is_module_loaded("libamdhip64.") {
        return;
    }

    tracing::debug!("Installing HIP hooks...");

    let install_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let mut hook_manager = HookManager::default();
        detour::mem::enable_hooks(&mut hook_manager)
    }));

    match install_result {
        Ok(Ok(())) => {
            HOOKS_INITIALIZED.store(true, Ordering::Release);
            tracing::debug!("HIP hooks installed successfully");
        }
        Ok(Err(error)) => {
            tracing::error!("HIP hooks installation failed: {error}");
        }
        Err(error) => {
            tracing::error!("HIP hooks installation panicked: {error:?}");
        }
    }
}

fn init_hooks() {
    if cfg!(test) {
        tracing::debug!("Test mode detected, skipping hook initialization");
        return;
    }

    init_limiter();

    let limiter = match GLOBAL_LIMITER.get() {
        Some(limiter) => limiter,
        None => {
            // Limiter failed to initialize (e.g., no hypervisor running).
            // Gracefully skip hooks — the library becomes a passthrough.
            report_limiter_not_initialized();
            return;
        }
    };

    let isolation = limiter.isolation();
    let should_skip_isolation = isolation.is_some_and(|iso| iso != "soft");

    if should_skip_isolation {
        tracing::info!(
            "Isolation level '{}' detected (non-soft), skipping hook initialization",
            isolation.expect("isolation checked above")
        );
        return;
    }

    let all_unlimited = GLOBAL_LIMITER
        .get()
        .map(|limiter| limiter.all_devices_unlimited())
        .unwrap_or(false);

    if should_skip_hooks_on_no_limit() && all_unlimited {
        tracing::info!("All devices have up_limit >= 100, skipping hooks installation");
        return;
    }

    // Try to install hooks immediately if libamdhip64 is already loaded
    if utils::hooks::is_module_loaded("libamdhip64.") {
        try_install_hip_hooks();
    }

    // Install dlsym hook to catch dynamic library loading
    static DLSYM_HOOK_ONCE: Once = Once::new();
    DLSYM_HOOK_ONCE.call_once(|| {
        let mut hook_manager = HookManager::default();
        if let Err(error) = replace_symbol!(
            &mut hook_manager,
            None,
            "dlsym",
            dlsym_detour,
            FnDlsym,
            FN_DLSYM
        ) {
            tracing::error!("Failed to install dlsym hook: {error}");
        }
    });
    tracing::debug!("Hook initialization completed");
}

thread_local! {
    static IN_DLSYM_DETOUR: Cell<bool> = const { Cell::new(false) };
}

fn call_original_dlsym(handle: *const c_void, symbol: *const c_char) -> *const c_void {
    if let Some(original) = FN_DLSYM.get() {
        unsafe { original(handle, symbol) }
    } else {
        unsafe { libc::dlsym(handle as *mut c_void, symbol) }
    }
}

#[hook_fn]
unsafe extern "C" fn dlsym_detour(handle: *const c_void, symbol: *const c_char) -> *const c_void {
    if symbol.is_null() {
        return call_original_dlsym(handle, symbol);
    }

    let Ok(symbol_str) = CStr::from_ptr(symbol).to_str() else {
        return call_original_dlsym(handle, symbol);
    };

    if !symbol_str.starts_with("hip") {
        return call_original_dlsym(handle, symbol);
    }

    // Prevent recursion
    if IN_DLSYM_DETOUR.with(|flag| flag.get()) {
        return call_original_dlsym(handle, symbol);
    }

    IN_DLSYM_DETOUR.with(|flag| flag.set(true));

    struct ResetGuard;
    impl Drop for ResetGuard {
        fn drop(&mut self) {
            IN_DLSYM_DETOUR.with(|flag| flag.set(false));
        }
    }
    let _guard = ResetGuard;

    if !HOOKS_INITIALIZED.load(Ordering::Acquire) {
        tracing::debug!("dlsym observed HIP symbol {symbol_str}, ensuring hooks installed");
        try_install_hip_hooks();
    }

    FN_DLSYM(handle, symbol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_remap_single_device_valid() {
        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("HIP_VISIBLE_DEVICES", "0");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("2".to_string()));

        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_single_device_out_of_range() {
        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("HIP_VISIBLE_DEVICES", "2");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Virtual device ID 2 out of range"));

        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_multiple_devices() {
        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("HIP_VISIBLE_DEVICES", "0,1");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("2,3".to_string()));

        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_no_original_env() {
        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("2,3".to_string()));

        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_empty_original_env() {
        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("HIP_VISIBLE_DEVICES", "");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("2,3".to_string()));

        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_invalid_device_id() {
        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("HIP_VISIBLE_DEVICES", "abc");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid device ID"));

        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_inherited_value_unchanged() {
        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("TF_REMAPPED", "2");
        env::set_var("HIP_VISIBLE_DEVICES", "2");
        let allocated = vec!["1".to_string(), "2".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("2".to_string()));

        env::remove_var("HIP_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    fn test_isolation_soft_should_not_skip() {
        let isolation = Some("soft");
        let should_skip = isolation.is_some_and(|iso| iso != "soft");
        assert!(!should_skip);
    }

    #[test]
    fn test_isolation_hard_should_skip() {
        let isolation = Some("hard");
        let should_skip = isolation.is_some_and(|iso| iso != "soft");
        assert!(should_skip);
    }

    #[test]
    fn test_isolation_none_should_not_skip() {
        let isolation: Option<&str> = None;
        let should_skip = isolation.is_some_and(|iso| iso != "soft");
        assert!(!should_skip);
    }
}
