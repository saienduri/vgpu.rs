use std::ffi::{c_int, c_uint, c_ulonglong, c_void};

use tf_macro::hook_fn;
use utils::hooks::HookManager;
use utils::replace_symbol;

use crate::hiplib::{HipDevice, HipError, HipMemPool, HipStream, HIP_ERROR_INVALID_VALUE,
                    HIP_ERROR_OUT_OF_MEMORY, HIP_ERROR_UNKNOWN, HIP_SUCCESS};
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

/// hipChannelFormatDesc — Runtime API channel format descriptor.
/// Fields x/y/z/w are bit widths per channel; f is hipChannelFormatKind (unused for sizing).
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct HipChannelFormatDesc {
    pub x: c_int,
    pub y: c_int,
    pub z: c_int,
    pub w: c_int,
    pub f: c_int,
}

/// HIP_ARRAY_DESCRIPTOR — Driver API 2D array descriptor.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct HipArrayDescriptor {
    pub width: usize,
    pub height: usize,
    pub format: c_int,
    pub num_channels: c_uint,
}

/// HIP_ARRAY3D_DESCRIPTOR — Driver API 3D array descriptor.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct HipArray3DDescriptor {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub format: c_int,
    pub num_channels: c_uint,
    pub flags: c_uint,
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

/// Bytes per element for a `hipArray_Format` enum value.
/// Returns `None` for unrecognized format values.
fn array_format_bytes(format: c_int) -> Option<u64> {
    match format {
        0x01 | 0x08 => Some(1), // UNSIGNED_INT8, SIGNED_INT8
        0x02 | 0x09 | 0x10 => Some(2), // UNSIGNED_INT16, SIGNED_INT16, HALF
        0x03 | 0x0a | 0x20 => Some(4), // UNSIGNED_INT32, SIGNED_INT32, FLOAT
        _ => None,
    }
}

/// Bytes per element from a `hipChannelFormatDesc` (sum of x/y/z/w bit widths / 8).
/// Returns `None` on negative widths, zero total bits, or non-byte-aligned bits.
fn channel_desc_bytes_per_elem(desc: &HipChannelFormatDesc) -> Option<u64> {
    if desc.x < 0 || desc.y < 0 || desc.z < 0 || desc.w < 0 {
        return None;
    }
    let total_bits = (desc.x as u64) + (desc.y as u64) + (desc.z as u64) + (desc.w as u64);
    if total_bits == 0 || total_bits % 8 != 0 {
        return None;
    }
    Some(total_bits / 8)
}

/// Compute allocation size for runtime API array descriptors (hipChannelFormatDesc).
/// Returns `None` on invalid descriptor, overflow, or size > u64::MAX/2.
fn channel_desc_alloc_size(
    desc: &HipChannelFormatDesc, width: usize, height: usize, depth: usize,
) -> Option<u64> {
    let bytes_per_elem = channel_desc_bytes_per_elem(desc)?;
    let h = if height == 0 { 1usize } else { height };
    let d = if depth == 0 { 1usize } else { depth };
    let size = bytes_per_elem
        .checked_mul(width as u64)?
        .checked_mul(h as u64)?
        .checked_mul(d as u64)?;
    if size <= u64::MAX / 2 { Some(size) } else { None }
}

/// Compute allocation size for driver API array descriptors (HIP_ARRAY_DESCRIPTOR / HIP_ARRAY3D_DESCRIPTOR).
/// Returns `None` on unknown format, overflow, or size > u64::MAX/2.
fn driver_array_alloc_size(
    format: c_int, num_channels: c_uint, width: usize, height: usize, depth: usize,
) -> Option<u64> {
    let elem_bytes = array_format_bytes(format)?;
    let h = if height == 0 { 1usize } else { height };
    let d = if depth == 0 { 1usize } else { depth };
    let size = elem_bytes
        .checked_mul(num_channels as u64)?
        .checked_mul(width as u64)?
        .checked_mul(h as u64)?
        .checked_mul(d as u64)?;
    if size <= u64::MAX / 2 { Some(size) } else { None }
}

/// Compute total allocation size for a mipmapped array by summing all mip levels.
///
/// Each level halves each dimension (floored to 1). This is more accurate than the
/// geometric series upper bound (2x for 1D, 4/3x for 2D, 8/7x for 3D) because it
/// uses the actual `num_levels` and integer-floored dimensions.
///
/// The naive per-level formula may slightly undercount vs the driver's internal
/// tiling/padding, but this is acceptable: mipmapped arrays are rare in ML workloads,
/// slight undercount favors the user, and no HIP API exists to query actual consumption.
///
/// `num_levels` is capped at 32 to prevent pathological iteration. The maximum
/// meaningful mip level count for the largest supported texture dimension (65536) is
/// `floor(log2(65536)) + 1 = 17`, so 32 is generous while still bounded.
fn mip_chain_total_size(
    bytes_per_elem: u64, width: usize, height: usize, depth: usize, num_levels: u32,
) -> Option<u64> {
    const MAX_MIP_LEVELS: u32 = 32;
    if num_levels > MAX_MIP_LEVELS {
        return None;
    }
    let mut total: u64 = 0;
    let mut w = width;
    let mut h = if height == 0 { 1usize } else { height };
    let mut d = if depth == 0 { 1usize } else { depth };
    for _ in 0..num_levels {
        let level_size = bytes_per_elem
            .checked_mul(w as u64)?
            .checked_mul(h as u64)?
            .checked_mul(d as u64)?;
        total = total.checked_add(level_size)?;
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        d = (d / 2).max(1);
    }
    if total <= u64::MAX / 2 { Some(total) } else { None }
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

// --- Virtual memory management hooks ---
//
// hipMemCreate/hipMemRelease use opaque handles (not device pointers) to track
// physical GPU memory allocations. The handle is stored in the same DashMap as
// device pointers — the keyspaces don't collide because handles are host-heap
// pointers while device pointers are GPU virtual addresses.

/// hipMemCreate allocates physical GPU memory backing, returning an opaque handle.
/// The size parameter is the allocation size in bytes (must be granularity-aligned).
/// We pass `prop` as opaque (*const c_void) since we only need `size` for accounting.
#[hook_fn]
pub(crate) unsafe extern "C" fn hip_mem_create_detour(
    handle: *mut *mut c_void,
    size: usize,
    prop: *const c_void,
    flags: c_ulonglong,
) -> HipError {
    let request_size = size as u64;
    check_and_alloc!(handle, request_size, "hipMemCreate", || {
        FN_HIP_MEM_CREATE(handle, size, prop, flags)
    })
}

/// hipMemRelease frees a physical GPU memory handle previously created by hipMemCreate.
///
/// Handles obtained via hipMemRetainAllocationHandle or hipMemImportFromShareableHandle
/// are intentionally untracked — the limiter only accounts for the original hipMemCreate.
/// A retain'd handle's release decrements the internal refcount but the tracker entry
/// belongs to the original handle, so the second release is a safe no-op (over-reports).
#[hook_fn]
pub(crate) unsafe extern "C" fn hip_mem_release_detour(
    handle: *mut c_void,
) -> HipError {
    check_and_free!(handle, FN_HIP_MEM_RELEASE(handle))
}

// --- Array allocation hooks ---
//
// Array allocs use descriptor structs instead of explicit size parameters.
// We compute the size from the descriptor and feed it into check_and_alloc!.
// hipArray_t is an opaque pointer (host-heap), stored in the same DashMap as
// device pointers — keyspaces don't collide.

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_malloc_array_detour(
    array: *mut *mut c_void, // hipArray_t*
    desc: *const HipChannelFormatDesc,
    width: usize,
    height: usize,
    flags: c_uint,
) -> HipError {
    let Some(request_size) = channel_desc_alloc_size(&*desc, width, height, 0) else {
        tracing::error!("hipMallocArray: invalid channel format descriptor");
        return HIP_ERROR_INVALID_VALUE;
    };
    check_and_alloc!(array, request_size, "hipMallocArray", || {
        FN_HIP_MALLOC_ARRAY(array, desc, width, height, flags)
    })
}

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_malloc_3d_array_detour(
    array: *mut *mut c_void, // hipArray_t*
    desc: *const HipChannelFormatDesc,
    extent: HipExtent,
    flags: c_uint,
) -> HipError {
    let Some(request_size) = channel_desc_alloc_size(&*desc, extent.width, extent.height, extent.depth) else {
        tracing::error!("hipMalloc3DArray: invalid channel format descriptor");
        return HIP_ERROR_INVALID_VALUE;
    };
    check_and_alloc!(array, request_size, "hipMalloc3DArray", || {
        FN_HIP_MALLOC_3D_ARRAY(array, desc, extent, flags)
    })
}

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_array_create_detour(
    array: *mut *mut c_void, // hipArray_t*
    desc: *const HipArrayDescriptor,
) -> HipError {
    let d = &*desc;
    let Some(request_size) = driver_array_alloc_size(d.format, d.num_channels, d.width, d.height, 0) else {
        tracing::error!("hipArrayCreate: invalid array descriptor (format=0x{:x})", d.format);
        return HIP_ERROR_INVALID_VALUE;
    };
    check_and_alloc!(array, request_size, "hipArrayCreate", || {
        FN_HIP_ARRAY_CREATE(array, desc)
    })
}

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_array_3d_create_detour(
    array: *mut *mut c_void, // hipArray_t*
    desc: *const HipArray3DDescriptor,
) -> HipError {
    let d = &*desc;
    let Some(request_size) = driver_array_alloc_size(d.format, d.num_channels, d.width, d.height, d.depth) else {
        tracing::error!("hipArray3DCreate: invalid 3D array descriptor (format=0x{:x})", d.format);
        return HIP_ERROR_INVALID_VALUE;
    };
    check_and_alloc!(array, request_size, "hipArray3DCreate", || {
        FN_HIP_ARRAY_3D_CREATE(array, desc)
    })
}

// --- Array free hooks ---

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_free_array_detour(
    array: *mut c_void, // hipArray_t
) -> HipError {
    check_and_free!(array, FN_HIP_FREE_ARRAY(array))
}

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_array_destroy_detour(
    array: *mut c_void, // hipArray_t
) -> HipError {
    check_and_free!(array, FN_HIP_ARRAY_DESTROY(array))
}

// --- Mipmapped array allocation hooks ---
//
// Mipmapped arrays allocate a chain of progressively smaller mip levels.
// We sum all levels for accurate accounting (each level halves dimensions, floored to 1).
// hipMipmappedArray_t is an opaque host-heap pointer (distinct heap allocation from the
// HIP runtime), so its address cannot collide with device VA pointers or other handle
// types in the shared DashMap tracker.

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_malloc_mipmapped_array_detour(
    array: *mut *mut c_void, // hipMipmappedArray_t*
    desc: *const HipChannelFormatDesc,
    extent: HipExtent, // passed by value
    num_levels: c_uint,
    flags: c_uint,
) -> HipError {
    let Some(bytes_per_elem) = channel_desc_bytes_per_elem(&*desc) else {
        tracing::error!("hipMallocMipmappedArray: invalid channel format descriptor");
        return HIP_ERROR_INVALID_VALUE;
    };
    let Some(request_size) = mip_chain_total_size(bytes_per_elem, extent.width, extent.height, extent.depth, num_levels) else {
        tracing::error!("hipMallocMipmappedArray: size computation overflow");
        return HIP_ERROR_INVALID_VALUE;
    };
    check_and_alloc!(array, request_size, "hipMallocMipmappedArray", || {
        FN_HIP_MALLOC_MIPMAPPED_ARRAY(array, desc, extent, num_levels, flags)
    })
}

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_mipmapped_array_create_detour(
    array: *mut *mut c_void, // hipMipmappedArray_t*
    desc: *mut HipArray3DDescriptor, // non-const pointer per HIP API
    num_levels: c_uint,
) -> HipError {
    let d = &*desc;
    let Some(elem_bytes) = array_format_bytes(d.format) else {
        tracing::error!("hipMipmappedArrayCreate: invalid array descriptor (format=0x{:x})", d.format);
        return HIP_ERROR_INVALID_VALUE;
    };
    let bytes_per_elem = match elem_bytes.checked_mul(d.num_channels as u64) {
        Some(b) => b,
        None => {
            tracing::error!("hipMipmappedArrayCreate: element size overflow (format=0x{:x}, num_channels={})", d.format, d.num_channels);
            return HIP_ERROR_INVALID_VALUE;
        }
    };
    let Some(request_size) = mip_chain_total_size(bytes_per_elem, d.width, d.height, d.depth, num_levels) else {
        tracing::error!("hipMipmappedArrayCreate: size computation overflow");
        return HIP_ERROR_INVALID_VALUE;
    };
    check_and_alloc!(array, request_size, "hipMipmappedArrayCreate", || {
        FN_HIP_MIPMAPPED_ARRAY_CREATE(array, desc, num_levels)
    })
}

// --- Mipmapped array free hooks ---

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_free_mipmapped_array_detour(
    array: *mut c_void, // hipMipmappedArray_t
) -> HipError {
    check_and_free!(array, FN_HIP_FREE_MIPMAPPED_ARRAY(array))
}

#[hook_fn]
pub(crate) unsafe extern "C" fn hip_mipmapped_array_destroy_detour(
    array: *mut c_void, // hipMipmappedArray_t
) -> HipError {
    check_and_free!(array, FN_HIP_MIPMAPPED_ARRAY_DESTROY(array))
}

// --- Info spoofing hooks ---

/// Partial repr(C) mirror of hipDeviceProp_t — only fields up to `totalGlobalMem`.
/// Used to patch the total memory field after calling the real hipGetDeviceProperties.
/// Layout from hip_runtime_api.h (ROCm 6.x):
///   char name[256]; hipUUID uuid; char luid[8]; unsigned int luidDeviceNodeMask;
///   size_t totalGlobalMem; ...
#[repr(C)]
struct HipDevicePropPrefix {
    name: [u8; 256],
    uuid: [u8; 16],   // hipUUID
    luid: [u8; 8],
    luid_device_node_mask: u32,
    // 4 bytes implicit padding (repr(C) aligns total_global_mem to 8 bytes)
    total_global_mem: usize,     // offset 288
}

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

/// Spoof `hipGetDeviceProperties` to report our memory limit as `totalGlobalMem`.
///
/// PyTorch's caching allocator reads `device_prop.totalGlobalMem` (from `hipGetDeviceProperties`,
/// not `hipMemGetInfo`) to compute the byte limit for `set_per_process_memory_fraction()`.
/// Without this hook, the fraction is computed against the real 256 GiB, producing a much larger
/// byte limit than intended and preventing OOM tests from triggering.
///
/// Strategy: call the real function first (to populate all ~80 fields), then patch only
/// `totalGlobalMem` with our spoofed limit.
#[hook_fn]
pub(crate) unsafe extern "C" fn hip_get_device_properties_detour(
    prop: *mut HipDevicePropPrefix,
    device_id: c_int,
) -> HipError {
    if prop.is_null() {
        return HIP_ERROR_INVALID_VALUE;
    }
    let result = FN_HIP_GET_DEVICE_PROPERTIES(prop, device_id);
    if result != HIP_SUCCESS {
        return result;
    }

    let limiter = match GLOBAL_LIMITER.get() {
        Some(limiter) => limiter,
        None => return result, // limiter not initialized, return unpatched
    };

    let device_idx = match limiter.device_index_by_hip_device(device_id) {
        Ok(idx) => idx,
        Err(_) => return result, // unmapped device, return unpatched
    };

    match limiter.get_pod_memory_usage(device_idx) {
        Ok((_used, limit)) => {
            (*prop).total_global_mem = limit as usize;
        }
        Err(e) => {
            tracing::warn!(device_id, ?e, "hipGetDeviceProperties: failed to get pod memory usage, returning unpatched");
        }
    }

    result
}

/// Spoof `hipGetDevicePropertiesR0600` — versioned variant called by PyTorch (compiled against ROCm 6.x).
/// Same ABI and struct layout as `hipGetDeviceProperties`; needs its own hook because Frida hooks
/// by address and the versioned symbols resolve to different entry points in libamdhip64.so.
#[hook_fn]
pub(crate) unsafe extern "C" fn hip_get_device_properties_r0600_detour(
    prop: *mut HipDevicePropPrefix,
    device_id: c_int,
) -> HipError {
    if prop.is_null() {
        return HIP_ERROR_INVALID_VALUE;
    }
    let result = FN_HIP_GET_DEVICE_PROPERTIES_R0600(prop, device_id);
    if result != HIP_SUCCESS {
        return result;
    }

    let limiter = match GLOBAL_LIMITER.get() {
        Some(limiter) => limiter,
        None => return result,
    };

    let device_idx = match limiter.device_index_by_hip_device(device_id) {
        Ok(idx) => idx,
        Err(_) => return result,
    };

    match limiter.get_pod_memory_usage(device_idx) {
        Ok((_used, limit)) => {
            (*prop).total_global_mem = limit as usize;
        }
        Err(e) => {
            tracing::warn!(device_id, ?e, "hipGetDevicePropertiesR0600: failed to get pod memory usage, returning unpatched");
        }
    }

    result
}

/// Spoof `hipGetDevicePropertiesR0000` — legacy versioned variant.
#[hook_fn]
pub(crate) unsafe extern "C" fn hip_get_device_properties_r0000_detour(
    prop: *mut HipDevicePropPrefix,
    device_id: c_int,
) -> HipError {
    if prop.is_null() {
        return HIP_ERROR_INVALID_VALUE;
    }
    let result = FN_HIP_GET_DEVICE_PROPERTIES_R0000(prop, device_id);
    if result != HIP_SUCCESS {
        return result;
    }

    let limiter = match GLOBAL_LIMITER.get() {
        Some(limiter) => limiter,
        None => return result,
    };

    let device_idx = match limiter.device_index_by_hip_device(device_id) {
        Ok(idx) => idx,
        Err(_) => return result,
    };

    match limiter.get_pod_memory_usage(device_idx) {
        Ok((_used, limit)) => {
            (*prop).total_global_mem = limit as usize;
        }
        Err(e) => {
            tracing::warn!(device_id, ?e, "hipGetDevicePropertiesR0000: failed to get pod memory usage, returning unpatched");
        }
    }

    result
}

/// Attaches Frida GUM hooks to all HIP memory allocation, deallocation, and info-spoofing APIs.
///
/// # Hook coverage (27 hooks registered here; 31 total including smi.rs and dlsym)
///
/// **Alloc (15):** hipMalloc, hipExtMallocWithFlags, hipMallocManaged, hipMallocAsync,
/// hipMallocFromPoolAsync, hipMallocPitch, hipMemAllocPitch, hipMalloc3D, hipMemCreate,
/// hipMallocArray, hipMalloc3DArray, hipArrayCreate, hipArray3DCreate, hipMallocMipmappedArray,
/// hipMipmappedArrayCreate
///
/// **Free (7):** hipFree, hipFreeAsync, hipMemRelease, hipFreeArray, hipArrayDestroy,
/// hipFreeMipmappedArray, hipMipmappedArrayDestroy
///
/// **Spoofing (5 here):** hipMemGetInfo, hipDeviceTotalMem, hipGetDeviceProperties{,R0600,R0000}
/// (3 more in smi.rs via dlsym: rsmi_dev_memory_total_get, amdsmi_get_gpu_memory_total,
/// amdsmi_get_gpu_vram_info; plus 1 dlsym hook in hip_limiter.rs)
///
/// # Known gap: graph memory nodes
///
/// `hipGraphAddMemAllocNode` and `hipGraphAddMemFreeNode` are NOT hooked. These APIs use an
/// internal CLR pool allocator (`MemoryPool::AllocateMemory` → `amd::SvmBuffer::malloc()`) that
/// bypasses all public HIP APIs — neither `hipMalloc` nor `hipFree` is called during graph
/// execution. Physical VRAM is allocated at `hipGraphLaunch` time, retained in a per-device pool
/// across graph lifetimes, and only released to the OS via `hipDeviceGraphMemTrim()`.
///
/// This gap is accepted because:
/// 1. No ML framework (PyTorch, JAX, TensorFlow, ONNX Runtime) uses graph memory alloc nodes —
///    all pre-allocate via private pools or arena allocators before capture. The one edge case is
///    cuBLAS/hipBLAS 12+ internal workspace calls during stream capture, which frameworks work
///    around by pre-setting workspace via `cublasSetWorkspace()`.
/// 2. No production GPU limiter (HAMi-core, tkestack/vcuda, NVIDIA MPS) hooks graph memory.
/// 3. Synchronous enforcement is impossible — the internal allocator has no public interception
///    point, and before/after queries on `hipDeviceGetGraphMemAttribute` are racy.
///
/// If graph memory becomes relevant, `hipDeviceGetGraphMemAttribute(hipGraphMemAttrReservedMemCurrent)`
/// can monitor pool usage, and `hipDeviceGraphMemTrim()` can reclaim inactive memory.
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
        "hipFreeAsync",
        hip_free_async_detour,
        FnHip_free_async,
        FN_HIP_FREE_ASYNC
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMemCreate",
        hip_mem_create_detour,
        FnHip_mem_create,
        FN_HIP_MEM_CREATE
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMemRelease",
        hip_mem_release_detour,
        FnHip_mem_release,
        FN_HIP_MEM_RELEASE
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
    // --- Array free hooks (registered before array alloc hooks for consistency) ---
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipFreeArray",
        hip_free_array_detour,
        FnHip_free_array,
        FN_HIP_FREE_ARRAY
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipArrayDestroy",
        hip_array_destroy_detour,
        FnHip_array_destroy,
        FN_HIP_ARRAY_DESTROY
    )?;
    // --- Array alloc hooks ---
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMallocArray",
        hip_malloc_array_detour,
        FnHip_malloc_array,
        FN_HIP_MALLOC_ARRAY
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMalloc3DArray",
        hip_malloc_3d_array_detour,
        FnHip_malloc_3d_array,
        FN_HIP_MALLOC_3D_ARRAY
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipArrayCreate",
        hip_array_create_detour,
        FnHip_array_create,
        FN_HIP_ARRAY_CREATE
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipArray3DCreate",
        hip_array_3d_create_detour,
        FnHip_array_3d_create,
        FN_HIP_ARRAY_3D_CREATE
    )?;
    // --- Mipmapped array free hooks (registered before mipmapped alloc hooks for consistency) ---
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipFreeMipmappedArray",
        hip_free_mipmapped_array_detour,
        FnHip_free_mipmapped_array,
        FN_HIP_FREE_MIPMAPPED_ARRAY
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMipmappedArrayDestroy",
        hip_mipmapped_array_destroy_detour,
        FnHip_mipmapped_array_destroy,
        FN_HIP_MIPMAPPED_ARRAY_DESTROY
    )?;
    // --- Mipmapped array alloc hooks ---
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMallocMipmappedArray",
        hip_malloc_mipmapped_array_detour,
        FnHip_malloc_mipmapped_array,
        FN_HIP_MALLOC_MIPMAPPED_ARRAY
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipMipmappedArrayCreate",
        hip_mipmapped_array_create_detour,
        FnHip_mipmapped_array_create,
        FN_HIP_MIPMAPPED_ARRAY_CREATE
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
    // hipGetDeviceProperties has three versioned symbols in libamdhip64.so, each at a
    // different address: hipGetDeviceProperties (default @@hip_4.2),
    // hipGetDevicePropertiesR0000 (@@hip_4.2), hipGetDevicePropertiesR0600 (@@hip_6.0).
    // PyTorch compiles against the R0600 variant. All three share the same ABI.
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipGetDeviceProperties",
        hip_get_device_properties_detour,
        FnHip_get_device_properties,
        FN_HIP_GET_DEVICE_PROPERTIES
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipGetDevicePropertiesR0600",
        hip_get_device_properties_r0600_detour,
        FnHip_get_device_properties_r0600,
        FN_HIP_GET_DEVICE_PROPERTIES_R0600
    )?;
    replace_symbol!(
        hook_manager,
        Some("libamdhip64."),
        "hipGetDevicePropertiesR0000",
        hip_get_device_properties_r0000_detour,
        FnHip_get_device_properties_r0000,
        FN_HIP_GET_DEVICE_PROPERTIES_R0000
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- array_format_bytes ---

    #[test]
    fn test_format_bytes_unsigned_int8() {
        assert_eq!(array_format_bytes(0x01), Some(1));
    }

    #[test]
    fn test_format_bytes_unsigned_int16() {
        assert_eq!(array_format_bytes(0x02), Some(2));
    }

    #[test]
    fn test_format_bytes_unsigned_int32() {
        assert_eq!(array_format_bytes(0x03), Some(4));
    }

    #[test]
    fn test_format_bytes_signed_int8() {
        assert_eq!(array_format_bytes(0x08), Some(1));
    }

    #[test]
    fn test_format_bytes_signed_int16() {
        assert_eq!(array_format_bytes(0x09), Some(2));
    }

    #[test]
    fn test_format_bytes_signed_int32() {
        assert_eq!(array_format_bytes(0x0a), Some(4));
    }

    #[test]
    fn test_format_bytes_half() {
        assert_eq!(array_format_bytes(0x10), Some(2));
    }

    #[test]
    fn test_format_bytes_float() {
        assert_eq!(array_format_bytes(0x20), Some(4));
    }

    #[test]
    fn test_format_bytes_unknown() {
        assert_eq!(array_format_bytes(0x00), None);
        assert_eq!(array_format_bytes(0x04), None);
        assert_eq!(array_format_bytes(0xFF), None);
    }

    // --- channel_desc_alloc_size ---

    fn make_desc(x: i32, y: i32, z: i32, w: i32) -> HipChannelFormatDesc {
        HipChannelFormatDesc { x: x as c_int, y: y as c_int, z: z as c_int, w: w as c_int, f: 0 }
    }

    #[test]
    fn test_channel_desc_rgba_float_2d() {
        // 4 channels x 32 bits = 16 bytes/elem, 256x256
        let desc = make_desc(32, 32, 32, 32);
        assert_eq!(channel_desc_alloc_size(&desc, 256, 256, 0), Some(16 * 256 * 256));
    }

    #[test]
    fn test_channel_desc_single_u8_2d() {
        let desc = make_desc(8, 0, 0, 0);
        assert_eq!(channel_desc_alloc_size(&desc, 1024, 512, 0), Some(1024 * 512));
    }

    #[test]
    fn test_channel_desc_1d_height_zero() {
        let desc = make_desc(32, 0, 0, 0);
        assert_eq!(channel_desc_alloc_size(&desc, 1000, 0, 0), Some(4 * 1000));
    }

    #[test]
    fn test_channel_desc_zero_width() {
        let desc = make_desc(8, 0, 0, 0);
        assert_eq!(channel_desc_alloc_size(&desc, 0, 100, 0), Some(0));
    }

    #[test]
    fn test_channel_desc_non_byte_aligned() {
        let desc = make_desc(7, 0, 0, 0);
        assert_eq!(channel_desc_alloc_size(&desc, 100, 100, 0), None);
    }

    #[test]
    fn test_channel_desc_zero_bits() {
        let desc = make_desc(0, 0, 0, 0);
        assert_eq!(channel_desc_alloc_size(&desc, 100, 100, 0), None);
    }

    #[test]
    fn test_channel_desc_overflow() {
        let desc = make_desc(32, 0, 0, 0);
        assert_eq!(channel_desc_alloc_size(&desc, usize::MAX, 2, 0), None);
    }

    #[test]
    fn test_channel_desc_3d() {
        let desc = make_desc(32, 0, 0, 0);
        // 4 bytes * 64 * 64 * 64 = 1 MiB
        assert_eq!(channel_desc_alloc_size(&desc, 64, 64, 64), Some(4 * 64 * 64 * 64));
    }

    #[test]
    fn test_channel_desc_negative_bits() {
        assert_eq!(channel_desc_alloc_size(&make_desc(-8, 16, 0, 0), 100, 100, 0), None);
    }

    // --- driver_array_alloc_size ---

    #[test]
    fn test_driver_2ch_float_2d() {
        // FLOAT=0x20 (4 bytes) * 2 channels * 512 * 512 = 2 MiB
        assert_eq!(driver_array_alloc_size(0x20, 2, 512, 512, 0), Some(4 * 2 * 512 * 512));
    }

    #[test]
    fn test_driver_4ch_u8_3d() {
        // U8=0x01 (1 byte) * 4 channels * 64^3 = 1 MiB
        assert_eq!(driver_array_alloc_size(0x01, 4, 64, 64, 64), Some(4 * 64 * 64 * 64));
    }

    #[test]
    fn test_driver_1d_height_zero() {
        assert_eq!(driver_array_alloc_size(0x01, 1, 1000, 0, 0), Some(1000));
    }

    #[test]
    fn test_driver_unknown_format() {
        assert_eq!(driver_array_alloc_size(0xFF, 1, 100, 100, 0), None);
    }

    #[test]
    fn test_driver_overflow() {
        assert_eq!(driver_array_alloc_size(0x20, 4, usize::MAX, 2, 0), None);
    }

    // --- mip_chain_total_size ---

    #[test]
    fn test_mip_single_level_equals_base() {
        // 1 level = just the base: 4 bytes * 256 * 256 = 256 KiB
        assert_eq!(mip_chain_total_size(4, 256, 256, 0, 1), Some(4 * 256 * 256));
    }

    #[test]
    fn test_mip_2d_two_levels() {
        // Level 0: 4 * 256 * 256 = 262144
        // Level 1: 4 * 128 * 128 = 65536
        // Total: 327680
        assert_eq!(mip_chain_total_size(4, 256, 256, 0, 2), Some(262144 + 65536));
    }

    #[test]
    fn test_mip_2d_full_chain() {
        // 256x256 with 9 levels (256 -> 1x1)
        // Sum: 4*(256*256 + 128*128 + 64*64 + 32*32 + 16*16 + 8*8 + 4*4 + 2*2 + 1*1)
        //    = 4*(65536 + 16384 + 4096 + 1024 + 256 + 64 + 16 + 4 + 1) = 4*87381 = 349524
        assert_eq!(mip_chain_total_size(4, 256, 256, 0, 9), Some(4 * 87381));
    }

    #[test]
    fn test_mip_1d() {
        // 1D: width=128, height=0 (treated as 1), 8 levels
        // 128 + 64 + 32 + 16 + 8 + 4 + 2 + 1 = 255
        assert_eq!(mip_chain_total_size(1, 128, 0, 0, 8), Some(255));
    }

    #[test]
    fn test_mip_3d() {
        // 3D: 8x8x8, 4 levels, 1 byte/elem
        // Level 0: 8*8*8=512, Level 1: 4*4*4=64, Level 2: 2*2*2=8, Level 3: 1*1*1=1
        assert_eq!(mip_chain_total_size(1, 8, 8, 8, 4), Some(512 + 64 + 8 + 1));
    }

    #[test]
    fn test_mip_zero_levels() {
        // 0 levels = no allocation
        assert_eq!(mip_chain_total_size(4, 256, 256, 0, 0), Some(0));
    }

    #[test]
    fn test_mip_dimensions_floor_to_one() {
        // 3x1 2D with 3 levels: Level 0: 3*1=3, Level 1: 1*1=1, Level 2: 1*1=1
        assert_eq!(mip_chain_total_size(1, 3, 1, 0, 3), Some(3 + 1 + 1));
    }

    #[test]
    fn test_mip_overflow() {
        assert_eq!(mip_chain_total_size(4, usize::MAX, 2, 0, 1), None);
    }

    #[test]
    fn test_mip_exceeds_max_alloc() {
        // Large but not overflow — exceeds u64::MAX / 2 guard
        assert_eq!(mip_chain_total_size(u64::MAX / 4, 4, 1, 0, 1), None);
    }

    #[test]
    fn test_mip_exceeds_max_levels() {
        // num_levels > 32 is rejected
        assert_eq!(mip_chain_total_size(4, 256, 256, 0, 33), None);
        // 32 is the max allowed
        assert!(mip_chain_total_size(4, 256, 256, 0, 32).is_some());
    }

    #[test]
    fn test_mip_non_power_of_two() {
        // 100x50, 3 levels: Level 0: 100*50=5000, Level 1: 50*25=1250, Level 2: 25*12=300
        assert_eq!(mip_chain_total_size(1, 100, 50, 0, 3), Some(5000 + 1250 + 300));
    }

    // --- channel_desc_bytes_per_elem ---

    #[test]
    fn test_bytes_per_elem_rgba_float() {
        let desc = make_desc(32, 32, 32, 32);
        assert_eq!(channel_desc_bytes_per_elem(&desc), Some(16));
    }

    #[test]
    fn test_bytes_per_elem_two_channel() {
        let desc = make_desc(16, 16, 0, 0);
        assert_eq!(channel_desc_bytes_per_elem(&desc), Some(4));
    }

    #[test]
    fn test_bytes_per_elem_negative() {
        assert_eq!(channel_desc_bytes_per_elem(&make_desc(-8, 16, 0, 0)), None);
    }

    #[test]
    fn test_bytes_per_elem_non_byte_aligned() {
        assert_eq!(channel_desc_bytes_per_elem(&make_desc(7, 0, 0, 0)), None);
    }

    #[test]
    fn test_bytes_per_elem_zero() {
        assert_eq!(channel_desc_bytes_per_elem(&make_desc(0, 0, 0, 0)), None);
    }
}
