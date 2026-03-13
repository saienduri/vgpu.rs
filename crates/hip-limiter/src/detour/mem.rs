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

/// hipPitchedPtr — FFI struct populated by hipMalloc3D.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct HipPitchedPtr {
    pub ptr: *mut c_void,
    pub pitch: usize,
    pub xsize: usize,
    pub ysize: usize,
}

/// hipExtent — FFI struct for 3D extent dimensions.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct HipExtent {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

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

/// Free-then-record: call the native free FIRST, then update accounting.
///
/// This ordering is conservative: if the native free succeeds but the process crashes
/// before record_free, pod_memory_used over-reports (safe — other pods see less
/// available, not more). The alternative (record_free first) risks under-reporting
/// if native free fails, which could allow overcommit.
///
/// $ptr: the *mut c_void pointer to free
/// $free_fn: expression that calls the native free function, returning HipError
macro_rules! check_and_free {
    ($ptr:expr, $free_fn:expr) => {{
        let result = $free_fn;
        if result == HIP_SUCCESS && !$ptr.is_null() {
            if let Some(limiter) = GLOBAL_LIMITER.get() {
                limiter.record_free($ptr as usize);
            }
        }
        result
    }};
}

/// Two-phase pitched allocation: reserve estimated, call native, reserve alignment overhead.
///
/// Shared implementation for hipMallocPitch, hipMemAllocPitch, and hipMalloc3D. The GPU allocates
/// `pitch * height [* depth]` bytes where `pitch >= width` due to alignment, but
/// we only know `pitch` after the native call returns.
///
/// Flow:
/// 1. Reserve `estimated_size` (width * height [* depth])
/// 2. Call native allocator → get actual `pitch`
/// 3. Compute `actual_size` from pitch via `$actual_size_fn`
/// 4. If `actual_size > estimated_size`, reserve the extra overhead
///    - If over limit: rollback everything, free native alloc via FN_HIP_FREE, return OOM
///    - Uses FN_HIP_FREE (not the hooked detour) because the pointer was never
///      record_allocation'd — the detour would try to record_free a non-existent entry.
/// 5. Record allocation with `actual_size`
///
/// `$alloc_name`: string label for logging
/// `$estimated_size`: pre-computed u64, already validated (non-zero, within MAX_ALLOC_SIZE)
/// `$native_call`: expression that calls the native allocator, returning HipError
/// `$out_ptr_expr`: expression yielding the allocated *mut c_void (e.g., `*ptr` or `(*pitched).ptr`)
/// `$actual_size_fn`: closure `|pitch: usize| -> Option<usize>` computing actual size from pitch
/// `$out_pitch_expr`: expression yielding the actual pitch (e.g., `*pitch` or `(*pitched).pitch`)
macro_rules! check_and_alloc_pitched {
    ($alloc_name:expr, $estimated_size:expr, $native_call:expr, $out_ptr_expr:expr, $out_pitch_expr:expr, $actual_size_fn:expr) => {{
        match with_device!() {
            Ok((limiter, device_idx)) => match limiter.try_reserve(device_idx, $estimated_size) {
                Ok(_previous_used) => 'alloc: {
                    let result = $native_call;

                    if result != HIP_SUCCESS {
                        limiter.rollback_reservation(device_idx, $estimated_size);
                        break 'alloc result;
                    }

                    let allocated_ptr = $out_ptr_expr as usize;
                    if allocated_ptr == 0 {
                        limiter.rollback_reservation(device_idx, $estimated_size);
                        break 'alloc result;
                    }

                    let actual_pitch = $out_pitch_expr;
                    let actual_size = match ($actual_size_fn)(actual_pitch) {
                        Some(size) => size as u64,
                        None => {
                            limiter.rollback_reservation(device_idx, $estimated_size);
                            FN_HIP_FREE($out_ptr_expr);
                            break 'alloc HIP_ERROR_OUT_OF_MEMORY;
                        }
                    };
                    let extra = actual_size.saturating_sub($estimated_size);

                    if extra > 0 && limiter.try_reserve(device_idx, extra).is_err() {
                        tracing::warn!(
                            "{}: pitch ({}) > width, actual size ({}) exceeds limit after alignment overhead — denying",
                            $alloc_name, actual_pitch, actual_size
                        );
                        limiter.rollback_reservation(device_idx, $estimated_size);
                        FN_HIP_FREE($out_ptr_expr);
                        break 'alloc HIP_ERROR_OUT_OF_MEMORY;
                    }

                    limiter.record_allocation(device_idx, allocated_ptr, actual_size);
                    result
                }
                Err(error) => handle_reserve_error(error, $alloc_name),
            },
            Err(error) => {
                tracing::warn!("Device context error: {error}, falling back to native {}", $alloc_name);
                $native_call
            }
        }
    }};
}

/// Compute and validate the estimated size for a pitched allocation.
///
/// Multiplies all dimensions via checked arithmetic, then applies the MAX_ALLOC_SIZE
/// guard (u64::MAX / 2) to prevent transient wrapping of the atomic counter.
/// Returns `None` if any dimension overflows or the result exceeds the guard.
fn checked_pitched_size(dims: &[usize]) -> Option<u64> {
    let size = dims.iter().copied().try_fold(1usize, usize::checked_mul)?;
    if size <= u64::MAX as usize / 2 {
        Some(size as u64)
    } else {
        None
    }
}

// --- Allocation hooks ---

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_malloc_detour(
    ptr: *mut *mut c_void,
    size: usize,
) -> HipError {
    let request_size = size as u64;
    check_and_alloc!(ptr, request_size, "hipMalloc", || {
        FN_HIP_MALLOC(ptr, size)
    })
}

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_ext_malloc_with_flags_detour(
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
pub(crate) unsafe extern "C" fn hip_host_malloc_detour(
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
pub(crate) unsafe extern "C" fn hip_malloc_managed_detour(
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
pub(crate) unsafe extern "C" fn hip_malloc_async_detour(
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
pub(crate) unsafe extern "C" fn hip_malloc_from_pool_async_detour(
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

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_host_alloc_detour(
    ptr: *mut *mut c_void,
    size: usize,
    flags: c_uint,
) -> HipError {
    let request_size = size as u64;
    check_and_alloc!(ptr, request_size, "hipHostAlloc", || {
        FN_HIP_HOST_ALLOC(ptr, size, flags)
    })
}

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_malloc_host_detour(
    ptr: *mut *mut c_void,
    size: usize,
) -> HipError {
    let request_size = size as u64;
    check_and_alloc!(ptr, request_size, "hipMallocHost", || {
        FN_HIP_MALLOC_HOST(ptr, size)
    })
}

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_mem_alloc_host_detour(
    ptr: *mut *mut c_void,
    size: usize,
) -> HipError {
    let request_size = size as u64;
    check_and_alloc!(ptr, request_size, "hipMemAllocHost", || {
        FN_HIP_MEM_ALLOC_HOST(ptr, size)
    })
}

// --- Pitched allocation hooks ---
//
// hipMallocPitch, hipMemAllocPitch, and hipMalloc3D use a two-phase reserve pattern via check_and_alloc_pitched!
// because the actual GPU allocation size depends on pitch alignment (pitch >= width), which
// is only known after the native call returns.

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_malloc_pitch_detour(
    ptr: *mut *mut c_void,
    pitch: *mut usize,
    width: usize,
    height: usize,
) -> HipError {
    let Some(estimated_size) = checked_pitched_size(&[width, height]) else {
        return HIP_ERROR_OUT_OF_MEMORY;
    };

    if estimated_size == 0 {
        return FN_HIP_MALLOC_PITCH(ptr, pitch, width, height);
    }

    check_and_alloc_pitched!(
        "hipMallocPitch",
        estimated_size,
        FN_HIP_MALLOC_PITCH(ptr, pitch, width, height),
        *ptr,
        *pitch,
        |actual_pitch: usize| actual_pitch.checked_mul(height)
    )
}

/// Driver API version of hipMallocPitch. Same two-phase pitched pattern;
/// elementSizeBytes influences pitch alignment but doesn't affect accounting.
#[hook_fn]
pub(crate) unsafe extern "C" fn hip_mem_alloc_pitch_detour(
    dptr: *mut *mut c_void,
    pitch: *mut usize,
    width_in_bytes: usize,
    height: usize,
    element_size_bytes: c_uint,
) -> HipError {
    let Some(estimated_size) = checked_pitched_size(&[width_in_bytes, height]) else {
        return HIP_ERROR_OUT_OF_MEMORY;
    };

    if estimated_size == 0 {
        return FN_HIP_MEM_ALLOC_PITCH(dptr, pitch, width_in_bytes, height, element_size_bytes);
    }

    check_and_alloc_pitched!(
        "hipMemAllocPitch",
        estimated_size,
        FN_HIP_MEM_ALLOC_PITCH(dptr, pitch, width_in_bytes, height, element_size_bytes),
        *dptr,
        *pitch,
        |actual_pitch: usize| actual_pitch.checked_mul(height)
    )
}

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_malloc_3d_detour(
    pitched_dev_ptr: *mut HipPitchedPtr,
    extent: HipExtent,
) -> HipError {
    let width = extent.width;
    let height = extent.height;
    let depth = extent.depth;

    let Some(estimated_size) = checked_pitched_size(&[width, height, depth]) else {
        return HIP_ERROR_OUT_OF_MEMORY;
    };

    if estimated_size == 0 {
        return FN_HIP_MALLOC_3D(pitched_dev_ptr, extent);
    }

    check_and_alloc_pitched!(
        "hipMalloc3D",
        estimated_size,
        FN_HIP_MALLOC_3D(pitched_dev_ptr, extent),
        (*pitched_dev_ptr).ptr,
        (*pitched_dev_ptr).pitch,
        |actual_pitch: usize| actual_pitch.checked_mul(height).and_then(|ph| ph.checked_mul(depth))
    )
}

// --- Free hooks ---

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_free_detour(ptr: *mut c_void) -> HipError {
    check_and_free!(ptr, FN_HIP_FREE(ptr))
}

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_host_free_detour(ptr: *mut c_void) -> HipError {
    check_and_free!(ptr, FN_HIP_HOST_FREE(ptr))
}

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_free_host_detour(ptr: *mut c_void) -> HipError {
    check_and_free!(ptr, FN_HIP_FREE_HOST(ptr))
}

// NOTE: hipFreeAsync defers the actual GPU memory release until stream completion,
// but we decrement pod_memory_used immediately. This is intentional: deferring the
// decrement would over-report usage to other pods, causing unnecessary OOM denials.
// If a subsequent hipMalloc fails because the GPU hasn't actually freed yet, the
// check_and_alloc! macro handles it correctly (rolls back the reservation).
#[hook_fn]
pub(crate) unsafe extern "C" fn hip_free_async_detour(
    ptr: *mut c_void,
    stream: HipStream,
) -> HipError {
    check_and_free!(ptr, FN_HIP_FREE_ASYNC(ptr, stream))
}

// --- Info spoofing hooks ---

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_mem_get_info_detour(
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
pub(crate) unsafe extern "C" fn hip_device_total_mem_detour(
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
        "hipHostAlloc",
        hip_host_alloc_detour,
        FnHip_host_alloc,
        FN_HIP_HOST_ALLOC
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMallocHost",
        hip_malloc_host_detour,
        FnHip_malloc_host,
        FN_HIP_MALLOC_HOST
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMemAllocHost",
        hip_mem_alloc_host_detour,
        FnHip_mem_alloc_host,
        FN_HIP_MEM_ALLOC_HOST
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
    // Free hooks must be registered before pitched alloc hooks (hipMallocPitch,
    // hipMemAllocPitch, hipMalloc3D) because check_and_alloc_pitched! calls FN_HIP_FREE on the
    // rollback path. Registration order doesn't affect runtime correctness (all
    // hooks are installed before any are invoked), but keeping this order makes
    // the dependency explicit for future maintainers.
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
        "hipFreeHost",
        hip_free_host_detour,
        FnHip_free_host,
        FN_HIP_FREE_HOST
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
        "hipMallocPitch",
        hip_malloc_pitch_detour,
        FnHip_malloc_pitch,
        FN_HIP_MALLOC_PITCH
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMemAllocPitch",
        hip_mem_alloc_pitch_detour,
        FnHip_mem_alloc_pitch,
        FN_HIP_MEM_ALLOC_PITCH
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMalloc3D",
        hip_malloc_3d_detour,
        FnHip_malloc_3d,
        FN_HIP_MALLOC_3D
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
