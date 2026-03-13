use std::cell::Cell;
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
pub(crate) mod detour;
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
        return;
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

fn init_limiter() {
    static LIMITER_INITIALIZED: Once = Once::new();
    LIMITER_INITIALIZED.call_once(|| {
        match hiplib::init_hiplib() {
            Ok(_) => {}
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

        // NOTE: Device visibility is the platform's responsibility (K8s device plugin),
        // not the limiter's. The limiter enforces memory limits via SHM hooks on whichever
        // GPUs are visible. We do not set HIP_VISIBLE_DEVICES here because the limiter's
        // #[ctor] initializes the HIP runtime (via hipGetDeviceCount) before we could set
        // it, and HIP only reads HIP_VISIBLE_DEVICES at first initialization.

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

    // Use Once to ensure only one thread attempts hook installation, even if
    // multiple threads race past the HOOKS_INITIALIZED fast-path check above.
    static INSTALL_ONCE: Once = Once::new();
    INSTALL_ONCE.call_once(|| {
        tracing::debug!("Installing HIP hooks...");

        let install_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            let mut hook_manager = HookManager::default();
            detour::mem::enable_hooks(&mut hook_manager)?;
            Ok::<(), utils::HookError>(())
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
    });
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

    let all_unlimited = limiter.all_devices_unlimited();

    if should_skip_hooks_on_no_limit() && all_unlimited {
        tracing::info!("All devices have up_limit >= 100, skipping hooks installation");
        return;
    }

    // Try to install hooks immediately if libraries are already loaded
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

    let is_hip_symbol = symbol_str.starts_with("hip");
    let is_smi_symbol = symbol_str.starts_with("rsmi_") || symbol_str.starts_with("amdsmi_");

    if !is_hip_symbol && !is_smi_symbol {
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

    if is_hip_symbol && !HOOKS_INITIALIZED.load(Ordering::Acquire) {
        tracing::debug!("dlsym observed HIP symbol {symbol_str}, ensuring hooks installed");
        try_install_hip_hooks();
    }

    // For SMI symbols, intercept at the dlsym level: resolve the original
    // address and return our detour's address. This avoids Frida reentrancy
    // issues when installing hooks from within the dlsym detour.
    let original = call_original_dlsym(handle, symbol);
    if is_smi_symbol && !original.is_null() {
        if let Some(detour) = detour::smi::try_intercept_smi_symbol(symbol_str, original) {
            return detour;
        }
    }

    original
}

