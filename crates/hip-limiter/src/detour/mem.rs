use std::ffi::c_uint;
use std::ffi::c_void;

use tf_macro::hook_fn;
use utils::hooks::HookManager;
use utils::replace_symbol;

use crate::hiplib::{HipDevice, HipError, HipMemPool, HipStream, HIP_ERROR_OUT_OF_MEMORY,
                    HIP_ERROR_UNKNOWN, HIP_SUCCESS};
use crate::limiter::Error;
use crate::with_device;
use crate::Limiter;
use crate::GLOBAL_LIMITER;

/// Check pod-level memory allocation and execute the allocation if within limits.
///
/// Returns the original allocation result on success, or hipErrorOutOfMemory if denied.
macro_rules! check_and_alloc {
    ($request_size:expr, $alloc_name:expr, $alloc_fn:expr) => {{
        let device_result = with_device!(|limiter: &crate::limiter::Limiter, device_idx: usize| {
            (limiter.get_pod_memory_usage(device_idx), device_idx)
        });
        match device_result {
            Ok((result, device_idx)) => match result {
                Ok((used, mem_limit)) if used.saturating_add($request_size) > mem_limit => {
                    tracing::warn!(
                        "Allocation denied by limiter ({}): used ({}) + request ({}) > limit ({}) device_idx: {}",
                        $alloc_name,
                        used,
                        $request_size,
                        mem_limit,
                        device_idx
                    );
                    HIP_ERROR_OUT_OF_MEMORY
                }
                Ok(_) => $alloc_fn(),
                Err(Error::DeviceNotHealthy { device_idx, last_heartbeat }) => {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    tracing::warn!(
                        now = now,
                        device_idx = device_idx,
                        last_heartbeat = last_heartbeat,
                        "Device not healthy, allowing allocation as fallback"
                    );
                    $alloc_fn()
                }
                Err(error) => {
                    tracing::error!("Failed to get pod memory usage: {error}");
                    HIP_ERROR_UNKNOWN
                }
            },
            Err(error) => {
                tracing::warn!("Device context error: {error}, falling back to native call");
                $alloc_fn()
            }
        }
    }};
}

// --- Allocation hooks ---

#[hook_fn]
pub(crate) unsafe fn hip_malloc_detour(
    ptr: *mut *mut c_void,
    size: usize,
) -> HipError {
    let request_size = size as u64;
    check_and_alloc!(request_size, "hipMalloc", || {
        FN_HIP_MALLOC(ptr, size)
    })
}

#[hook_fn]
pub(crate) unsafe fn hip_ext_malloc_with_flags_detour(
    ptr: *mut *mut c_void,
    size_bytes: usize,
    flags: c_uint,
) -> HipError {
    let request_size = size_bytes as u64;
    check_and_alloc!(request_size, "hipExtMallocWithFlags", || {
        FN_HIP_EXT_MALLOC_WITH_FLAGS(ptr, size_bytes, flags)
    })
}

#[hook_fn]
pub(crate) unsafe fn hip_host_malloc_detour(
    ptr: *mut *mut c_void,
    size: usize,
    flags: c_uint,
) -> HipError {
    let request_size = size as u64;
    check_and_alloc!(request_size, "hipHostMalloc", || {
        FN_HIP_HOST_MALLOC(ptr, size, flags)
    })
}

#[hook_fn]
pub(crate) unsafe fn hip_malloc_managed_detour(
    dev_ptr: *mut *mut c_void,
    size: usize,
    flags: c_uint,
) -> HipError {
    let request_size = size as u64;
    check_and_alloc!(request_size, "hipMallocManaged", || {
        FN_HIP_MALLOC_MANAGED(dev_ptr, size, flags)
    })
}

#[hook_fn]
pub(crate) unsafe fn hip_malloc_async_detour(
    dev_ptr: *mut *mut c_void,
    size: usize,
    stream: HipStream,
) -> HipError {
    let request_size = size as u64;
    check_and_alloc!(request_size, "hipMallocAsync", || {
        FN_HIP_MALLOC_ASYNC(dev_ptr, size, stream)
    })
}

#[hook_fn]
pub(crate) unsafe fn hip_malloc_from_pool_async_detour(
    dev_ptr: *mut *mut c_void,
    size: usize,
    mem_pool: HipMemPool,
    stream: HipStream,
) -> HipError {
    let request_size = size as u64;
    check_and_alloc!(request_size, "hipMallocFromPoolAsync", || {
        FN_HIP_MALLOC_FROM_POOL_ASYNC(dev_ptr, size, mem_pool, stream)
    })
}

#[hook_fn]
pub(crate) unsafe fn hip_malloc_pitch_detour(
    ptr: *mut *mut c_void,
    pitch: *mut usize,
    width: usize,
    height: usize,
) -> HipError {
    let request_size = (width * height) as u64;
    check_and_alloc!(request_size, "hipMallocPitch", || {
        FN_HIP_MALLOC_PITCH(ptr, pitch, width, height)
    })
}

// --- Info spoofing hooks ---

#[hook_fn]
pub(crate) unsafe fn hip_mem_get_info_detour(
    free: *mut usize,
    total: *mut usize,
) -> HipError {
    let result = with_device!(|limiter: &Limiter, device_idx: usize| {
        match limiter.get_pod_memory_usage(device_idx) {
            Ok((used, mem_limit)) => {
                *total = mem_limit as usize;
                *free = mem_limit.saturating_sub(used) as usize;
                HIP_SUCCESS
            }
            Err(Error::DeviceNotHealthy {
                device_idx,
                last_heartbeat,
            }) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                tracing::warn!(
                    now = now,
                    device_idx = device_idx,
                    last_heartbeat = last_heartbeat,
                    "Device not healthy"
                );
                HIP_ERROR_UNKNOWN
            }
            Err(error) => {
                tracing::error!("Failed to get pod memory usage: {error}");
                HIP_ERROR_UNKNOWN
            }
        }
    });

    match result {
        Ok(hip_result) => hip_result,
        Err(error) => {
            tracing::warn!("Device context error: {error}, falling back to native call");
            FN_HIP_MEM_GET_INFO(free, total)
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn hip_device_total_mem_detour(
    bytes: *mut usize,
    device: HipDevice,
) -> HipError {
    let limiter = match GLOBAL_LIMITER.get() {
        Some(limiter) => limiter,
        None => {
            report_limiter_not_initialized();
            return FN_HIP_DEVICE_TOTAL_MEM(bytes, device);
        }
    };

    match limiter.device_index_by_hip_device(device) {
        Ok(device_idx) => match limiter.get_pod_memory_usage(device_idx) {
            Ok((_used, limit)) => {
                *bytes = limit as usize;
                HIP_SUCCESS
            }
            Err(error) => {
                tracing::error!("Failed to get pod memory usage: {error}");
                HIP_ERROR_UNKNOWN
            }
        },
        Err(error) => {
            tracing::warn!("Device mapping error: {error}, falling back to native call");
            FN_HIP_DEVICE_TOTAL_MEM(bytes, device)
        }
    }
}

fn report_limiter_not_initialized() {
    crate::report_limiter_not_initialized();
}

pub(crate) unsafe fn enable_hooks(hook_manager: &mut HookManager) -> Result<(), utils::HookError> {
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMalloc",
        hip_malloc_detour,
        FnHip_malloc,
        FN_HIP_MALLOC
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipExtMallocWithFlags",
        hip_ext_malloc_with_flags_detour,
        FnHip_ext_malloc_with_flags,
        FN_HIP_EXT_MALLOC_WITH_FLAGS
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipHostMalloc",
        hip_host_malloc_detour,
        FnHip_host_malloc,
        FN_HIP_HOST_MALLOC
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMallocManaged",
        hip_malloc_managed_detour,
        FnHip_malloc_managed,
        FN_HIP_MALLOC_MANAGED
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMallocAsync",
        hip_malloc_async_detour,
        FnHip_malloc_async,
        FN_HIP_MALLOC_ASYNC
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMallocFromPoolAsync",
        hip_malloc_from_pool_async_detour,
        FnHip_malloc_from_pool_async,
        FN_HIP_MALLOC_FROM_POOL_ASYNC
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMallocPitch",
        hip_malloc_pitch_detour,
        FnHip_malloc_pitch,
        FN_HIP_MALLOC_PITCH
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMemGetInfo",
        hip_mem_get_info_detour,
        FnHip_mem_get_info,
        FN_HIP_MEM_GET_INFO
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipDeviceTotalMem",
        hip_device_total_mem_detour,
        FnHip_device_total_mem,
        FN_HIP_DEVICE_TOTAL_MEM
    )?;

    Ok(())
}
