use std::ffi::c_uint;
use std::ffi::c_void;

use tf_macro::hook_fn;
use utils::hooks::HookManager;
use utils::replace_symbol;

use crate::hiplib::{HipDevice, HipError, HipMemPool, HipStream, HIP_ERROR_OUT_OF_MEMORY,
                    HIP_ERROR_UNKNOWN, HIP_SUCCESS};
use crate::limiter::Error;
use crate::with_device;
use crate::GLOBAL_LIMITER;

/// Map a reserve error to a HIP error code.
fn handle_reserve_error(error: Error, alloc_name: &str) -> HipError {
    match error {
        Error::OverLimit { used, request, limit, device_idx } => {
            tracing::warn!(
                "Allocation denied by limiter ({}): used ({}) + request ({}) > limit ({}) device_idx: {}",
                alloc_name, used, request, limit, device_idx
            );
            HIP_ERROR_OUT_OF_MEMORY
        }
        error => {
            tracing::error!("Failed to reserve memory for {}: {error}", alloc_name);
            HIP_ERROR_UNKNOWN
        }
    }
}

/// Reserve-then-allocate: atomically reserves memory in SHM before calling the native
/// allocator, eliminating the TOCTOU race in the old check-then-allocate pattern.
///
/// Flow:
/// 1. Atomically increment pod_memory_used (reserve)
/// 2. If over limit → roll back, return OOM
/// 3. Call native allocator
/// 4. If native fails → roll back reservation
/// 5. Record pointer in tracker
///
/// $out_ptr: the *mut *mut c_void that receives the allocated pointer
/// $request_size: allocation size in bytes (u64)
/// $alloc_name: string label for logging
/// $alloc_fn: closure that calls the native allocation function
macro_rules! check_and_alloc {
    ($out_ptr:expr, $request_size:expr, $alloc_name:expr, $alloc_fn:expr) => {{
        match with_device!() {
            Ok((limiter, device_idx)) => match limiter.try_reserve(device_idx, $request_size) {
                Ok(_previous_used) => {
                    // Reservation succeeded — call the native allocator
                    let result = $alloc_fn();
                    if result == HIP_SUCCESS && $request_size > 0 {
                        let allocated_ptr = *$out_ptr as usize;
                        if allocated_ptr != 0 {
                            limiter.record_allocation(device_idx, allocated_ptr, $request_size);
                        } else {
                            // Native allocator returned success but null pointer — roll back reservation
                            limiter.rollback_reservation(device_idx, $request_size);
                        }
                    } else if result != HIP_SUCCESS && $request_size > 0 {
                        // Native alloc failed — roll back the reservation
                        limiter.rollback_reservation(device_idx, $request_size);
                    }
                    result
                }
                Err(error) => handle_reserve_error(error, $alloc_name),
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
    check_and_alloc!(ptr, request_size, "hipMalloc", || {
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
    check_and_alloc!(ptr, request_size, "hipExtMallocWithFlags", || {
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
    check_and_alloc!(ptr, request_size, "hipHostMalloc", || {
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
    check_and_alloc!(dev_ptr, request_size, "hipMallocManaged", || {
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
    check_and_alloc!(dev_ptr, request_size, "hipMallocAsync", || {
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
    check_and_alloc!(dev_ptr, request_size, "hipMallocFromPoolAsync", || {
        FN_HIP_MALLOC_FROM_POOL_ASYNC(dev_ptr, size, mem_pool, stream)
    })
}

/// hipMallocPitch uses a custom flow instead of check_and_alloc! because the GPU
/// allocates `pitch * height` bytes (where `pitch >= width` due to alignment), but
/// we only know `pitch` after the native call returns. The flow:
///
/// 1. Reserve `width * height` (the user-requested size)
/// 2. Call native hipMallocPitch → get actual `pitch`
/// 3. If `pitch > width`, reserve the extra `(pitch - width) * height`
///    - If that pushes over limit: rollback everything, free native alloc, return OOM
/// 4. Record allocation with actual size `pitch * height`
#[hook_fn]
pub(crate) unsafe fn hip_malloc_pitch_detour(
    ptr: *mut *mut c_void,
    pitch: *mut usize,
    width: usize,
    height: usize,
) -> HipError {
    let estimated_size = match width.checked_mul(height) {
        Some(size) => size as u64,
        None => return HIP_ERROR_OUT_OF_MEMORY,
    };

    match with_device!() {
        Ok((limiter, device_idx)) => match limiter.try_reserve(device_idx, estimated_size) {
            Ok(_previous_used) => {
                // Step 2: Call native allocator
                let result = FN_HIP_MALLOC_PITCH(ptr, pitch, width, height);

                if result != HIP_SUCCESS || estimated_size == 0 {
                    if estimated_size > 0 {
                        limiter.rollback_reservation(device_idx, estimated_size);
                    }
                    return result;
                }

                let allocated_ptr = *ptr as usize;
                if allocated_ptr == 0 {
                    limiter.rollback_reservation(device_idx, estimated_size);
                    return result;
                }

                // Step 3: Check actual pitch and reserve the alignment overhead
                let actual_pitch = *pitch;
                let actual_size = match actual_pitch.checked_mul(height) {
                    Some(size) => size as u64,
                    None => {
                        // Overflow — roll back and free
                        limiter.rollback_reservation(device_idx, estimated_size);
                        FN_HIP_FREE(*ptr);
                        return HIP_ERROR_OUT_OF_MEMORY;
                    }
                };
                let extra = actual_size.saturating_sub(estimated_size);

                if extra > 0 {
                    if let Err(_) = limiter.try_reserve(device_idx, extra) {
                        // Actual size exceeds limit — roll back everything and free
                        tracing::warn!(
                            "hipMallocPitch: pitch ({}) > width ({}), actual size ({}) exceeds limit after alignment overhead — denying",
                            actual_pitch, width, actual_size
                        );
                        limiter.rollback_reservation(device_idx, estimated_size);
                        FN_HIP_FREE(*ptr);
                        return HIP_ERROR_OUT_OF_MEMORY;
                    }
                }

                // Step 4: Record with actual size (pitch * height)
                limiter.record_allocation(device_idx, allocated_ptr, actual_size);
                result
            }
            Err(error) => handle_reserve_error(error, "hipMallocPitch"),
        },
        Err(error) => {
            tracing::warn!("Device context error: {error}, falling back to native hipMallocPitch");
            FN_HIP_MALLOC_PITCH(ptr, pitch, width, height)
        }
    }
}

// --- Free hooks ---

// Free hooks call the native free FIRST, then update accounting. This ordering is
// conservative: if the native free succeeds but the process crashes before record_free,
// pod_memory_used over-reports (safe — other pods see less available, not more). The
// alternative (record_free first) risks under-reporting if native free fails, which
// could allow overcommit.
#[hook_fn]
pub(crate) unsafe fn hip_free_detour(ptr: *mut c_void) -> HipError {
    let result = FN_HIP_FREE(ptr);
    if result == HIP_SUCCESS && !ptr.is_null() {
        if let Some(limiter) = GLOBAL_LIMITER.get() {
            limiter.record_free(ptr as usize);
        }
    }
    result
}

#[hook_fn]
pub(crate) unsafe fn hip_host_free_detour(ptr: *mut c_void) -> HipError {
    let result = FN_HIP_HOST_FREE(ptr);
    if result == HIP_SUCCESS && !ptr.is_null() {
        if let Some(limiter) = GLOBAL_LIMITER.get() {
            limiter.record_free(ptr as usize);
        }
    }
    result
}

// NOTE: hipFreeAsync defers the actual GPU memory release until stream completion,
// but we decrement pod_memory_used immediately. This is intentional: deferring the
// decrement would over-report usage to other pods, causing unnecessary OOM denials.
// If a subsequent hipMalloc fails because the GPU hasn't actually freed yet, the
// check_and_alloc! macro handles it correctly (rolls back the reservation).
#[hook_fn]
pub(crate) unsafe fn hip_free_async_detour(
    ptr: *mut c_void,
    stream: HipStream,
) -> HipError {
    let result = FN_HIP_FREE_ASYNC(ptr, stream);
    if result == HIP_SUCCESS && !ptr.is_null() {
        if let Some(limiter) = GLOBAL_LIMITER.get() {
            limiter.record_free(ptr as usize);
        }
    }
    result
}

// --- Info spoofing hooks ---

#[hook_fn]
pub(crate) unsafe fn hip_mem_get_info_detour(
    free: *mut usize,
    total: *mut usize,
) -> HipError {
    match with_device!() {
        Ok((limiter, device_idx)) => match limiter.get_pod_memory_usage(device_idx) {
            Ok((used, mem_limit)) => {
                *total = mem_limit as usize;
                *free = mem_limit.saturating_sub(used) as usize;
                HIP_SUCCESS
            }
            Err(error) => {
                tracing::error!("Failed to get pod memory usage: {error}");
                HIP_ERROR_UNKNOWN
            }
        },
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
            crate::report_limiter_not_initialized();
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
        "hipFree",
        hip_free_detour,
        FnHip_free,
        FN_HIP_FREE
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipHostFree",
        hip_host_free_detour,
        FnHip_host_free,
        FN_HIP_HOST_FREE
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipFreeAsync",
        hip_free_async_detour,
        FnHip_free_async,
        FN_HIP_FREE_ASYNC
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
