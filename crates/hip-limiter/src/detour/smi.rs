use std::ffi::c_uint;
use std::ffi::c_void;
use std::sync::OnceLock;

use dashmap::DashMap;
use tf_macro::hook_fn;

use crate::GLOBAL_LIMITER;

// --- rocm_smi FFI types ---

type RsmiStatus = u32;
type RsmiMemoryType = u32;

const RSMI_STATUS_SUCCESS: RsmiStatus = 0;
const RSMI_MEM_TYPE_VRAM: RsmiMemoryType = 0;

/// Resolve an rsmi device index to a limiter SHM device index.
///
/// rsmi and HIP both enumerate devices via KFD in the same order,
/// so rsmi device N = HIP device N.
fn resolve_rsmi_device(dv_ind: c_uint) -> Option<usize> {
    let limiter = GLOBAL_LIMITER.get()?;
    limiter.device_index_by_hip_device(dv_ind as i32).ok()
}

// --- rocm_smi hooks ---

/// Hook for rsmi_dev_memory_total_get — spoofs VRAM total to mem_limit.
/// Used by `rocm-smi --showmeminfo vram` and programmatic queries.
#[hook_fn]
pub(crate) unsafe extern "C" fn rsmi_dev_memory_total_get_detour(
    dv_ind: c_uint,
    mem_type: RsmiMemoryType,
    total: *mut u64,
) -> RsmiStatus {
    // Only spoof VRAM; pass through GTT, VIS_VRAM, etc.
    if mem_type != RSMI_MEM_TYPE_VRAM {
        return FN_RSMI_DEV_MEMORY_TOTAL_GET(dv_ind, mem_type, total);
    }

    let Some(device_idx) = resolve_rsmi_device(dv_ind) else {
        return FN_RSMI_DEV_MEMORY_TOTAL_GET(dv_ind, mem_type, total);
    };

    let limiter = GLOBAL_LIMITER.get().expect("checked in resolve_rsmi_device");
    match limiter.get_pod_memory_usage(device_idx) {
        Ok((_used, mem_limit)) => {
            *total = mem_limit;
            RSMI_STATUS_SUCCESS
        }
        Err(error) => {
            tracing::error!("Failed to get pod memory usage: {error}");
            FN_RSMI_DEV_MEMORY_TOTAL_GET(dv_ind, mem_type, total)
        }
    }
}

// --- amdsmi FFI types ---

type AmdSmiStatus = u32;
type AmdSmiProcessorHandle = *mut c_void;
type AmdSmiMemoryType = u32;

const AMDSMI_STATUS_SUCCESS: AmdSmiStatus = 0;
const AMDSMI_MEM_TYPE_VRAM: AmdSmiMemoryType = 0;

const AMDSMI_MAX_STRING_LENGTH: usize = 256;

/// amdsmi_vram_info_t — matches the C struct layout.
/// See amdsmi.h amdsmi_vram_info_t (refs/rocm-systems/projects/amdsmi/include/amd_smi/amdsmi.h).
/// We only need to overwrite `vram_size`; other fields are passed through.
#[repr(C)]
struct AmdSmiVramInfo {
    vram_type: u32, // amdsmi_vram_type_t (enum -> u32)
    vram_vendor: [u8; AMDSMI_MAX_STRING_LENGTH],
    vram_size: u64,          // in MB
    vram_bit_width: u32,     // in bits
    vram_max_bandwidth: u64, // in GB/s
    reserved: [u64; 37],
}

// --- amdsmi device resolution ---

/// amdsmi_get_gpu_device_bdf signature. Returns a packed BDF bitfield:
/// bits [2:0] = function, [7:3] = device, [15:8] = bus, [63:16] = domain.
type FnAmdSmiGetGpuDeviceBdf =
    unsafe extern "C" fn(AmdSmiProcessorHandle, *mut u64) -> AmdSmiStatus;

/// Cached function pointer for amdsmi_get_gpu_device_bdf, resolved from the
/// same library that provides the hooked amdsmi functions.
static FN_AMDSMI_GET_BDF: OnceLock<Option<FnAmdSmiGetGpuDeviceBdf>> = OnceLock::new();

/// Cache: processor handle pointer → SHM device index. Handles are stable for
/// the lifetime of the amdsmi session, so we resolve each handle at most once.
static AMDSMI_HANDLE_CACHE: OnceLock<DashMap<usize, Option<usize>>> = OnceLock::new();

/// Resolve amdsmi_get_gpu_device_bdf from the same library as `known_addr`.
///
/// Python ctypes loads libamd_smi with RTLD_LOCAL, so RTLD_DEFAULT can't find
/// symbols in it. Instead, we use dladdr on a known amdsmi function pointer to
/// find the library path, then dlopen+dlsym with that specific handle.
unsafe fn resolve_bdf_fn_from_lib(known_addr: *const c_void) -> Option<FnAmdSmiGetGpuDeviceBdf> {
    let mut info: libc::Dl_info = std::mem::zeroed();
    if libc::dladdr(known_addr, &mut info) == 0 || info.dli_fname.is_null() {
        tracing::warn!("dladdr failed on amdsmi function pointer");
        return None;
    }

    // Get a handle to the already-loaded library (RTLD_NOLOAD avoids reloading)
    let handle = libc::dlopen(info.dli_fname, libc::RTLD_NOW | libc::RTLD_NOLOAD);
    if handle.is_null() {
        let fname = std::ffi::CStr::from_ptr(info.dli_fname).to_string_lossy();
        tracing::warn!(lib = %fname, "dlopen(NOLOAD) failed for amdsmi library");
        return None;
    }

    let ptr = crate::real_dlsym(handle, c"amdsmi_get_gpu_device_bdf".as_ptr());
    // Don't dlclose — on glibc, dlopen(RTLD_NOLOAD) increments the refcount,
    // so dlclose would decrement it. Omitting dlclose leaves the refcount
    // unmodified, which is safe since the caller (Python) holds the library loaded.

    if ptr.is_null() {
        tracing::warn!("amdsmi_get_gpu_device_bdf not found in amdsmi library");
        None
    } else {
        tracing::debug!("resolved amdsmi_get_gpu_device_bdf at {ptr:?}");
        Some(std::mem::transmute(ptr))
    }
}

fn get_bdf_fn() -> Option<FnAmdSmiGetGpuDeviceBdf> {
    // Use get() instead of get_or_init(|| None) to avoid permanently freezing
    // the OnceLock to None if called before try_intercept_smi_symbol sets it.
    FN_AMDSMI_GET_BDF.get().copied().flatten()
}

/// Format an amdsmi BDF bitfield as a PCI bus ID string (e.g., "0000:75:00.0").
///
/// amdsmi_bdf_t layout: function(3) | device(5) | bus(8) | domain(48)
pub(crate) fn format_amdsmi_bdf(raw: u64) -> String {
    let function = raw & 0x7;
    let device = (raw >> 3) & 0x1F;
    let bus = (raw >> 8) & 0xFF;
    let domain = (raw >> 16) & 0xFFFFFFFFFFFF;
    format!("{domain:04x}:{bus:02x}:{device:02x}.{function}")
}

/// Resolve an amdsmi processor handle to a limiter SHM device index.
///
/// Uses amdsmi_get_gpu_device_bdf to get the PCI BDF from the opaque handle,
/// then matches it against the pod's configured GPU UUIDs (which are BDF-based).
/// Results are cached since handles are stable for the amdsmi session lifetime.
fn resolve_amdsmi_device(handle: AmdSmiProcessorHandle) -> Option<usize> {
    let handle_key = handle as usize;
    let cache = AMDSMI_HANDLE_CACHE.get_or_init(DashMap::new);

    // Fast path: already resolved this handle
    if let Some(cached) = cache.get(&handle_key) {
        return *cached;
    }

    // Get BDF from the processor handle
    let bdf_fn = get_bdf_fn()?;
    let mut bdf_raw: u64 = 0;
    let status = unsafe { bdf_fn(handle, &mut bdf_raw) };
    if status != AMDSMI_STATUS_SUCCESS {
        tracing::warn!("amdsmi_get_gpu_device_bdf failed: status={status}");
        cache.insert(handle_key, None);
        return None;
    }

    let bdf = format_amdsmi_bdf(bdf_raw);
    let limiter = GLOBAL_LIMITER.get()?;
    let result = limiter.device_index_by_pci_bdf(&bdf).ok();

    tracing::debug!(
        handle_key,
        bdf_raw,
        bdf,
        result = ?result,
        "amdsmi handle resolved via BDF"
    );

    cache.insert(handle_key, result);
    result
}

// --- amdsmi hooks ---

/// Hook for amdsmi_get_gpu_memory_total — spoofs VRAM total to mem_limit.
/// Used by `amd-smi monitor` and programmatic queries.
#[hook_fn]
pub(crate) unsafe extern "C" fn amdsmi_get_gpu_memory_total_detour(
    handle: AmdSmiProcessorHandle,
    mem_type: AmdSmiMemoryType,
    total: *mut u64,
) -> AmdSmiStatus {
    // Only spoof VRAM; pass through VIS_VRAM, GTT, etc.
    if mem_type != AMDSMI_MEM_TYPE_VRAM {
        return FN_AMDSMI_GET_GPU_MEMORY_TOTAL(handle, mem_type, total);
    }

    let Some(device_idx) = resolve_amdsmi_device(handle) else {
        return FN_AMDSMI_GET_GPU_MEMORY_TOTAL(handle, mem_type, total);
    };

    let limiter = GLOBAL_LIMITER.get().expect("checked in resolve_amdsmi_device");
    match limiter.get_pod_memory_usage(device_idx) {
        Ok((_used, mem_limit)) => {
            *total = mem_limit;
            AMDSMI_STATUS_SUCCESS
        }
        Err(error) => {
            tracing::error!("Failed to get pod memory usage: {error}");
            FN_AMDSMI_GET_GPU_MEMORY_TOTAL(handle, mem_type, total)
        }
    }
}

/// Hook for amdsmi_get_gpu_vram_info — spoofs vram_size to mem_limit in MB.
/// Used by `amd-smi static --vram`.
#[hook_fn]
pub(crate) unsafe extern "C" fn amdsmi_get_gpu_vram_info_detour(
    handle: AmdSmiProcessorHandle,
    info: *mut AmdSmiVramInfo,
) -> AmdSmiStatus {
    // Call real function first to populate all fields
    let status = FN_AMDSMI_GET_GPU_VRAM_INFO(handle, info);
    if status != AMDSMI_STATUS_SUCCESS {
        return status;
    }

    let Some(device_idx) = resolve_amdsmi_device(handle) else {
        return status;
    };

    let limiter = GLOBAL_LIMITER.get().expect("checked in resolve_amdsmi_device");
    match limiter.get_pod_memory_usage(device_idx) {
        Ok((_used, mem_limit)) => {
            // vram_size is in MB, mem_limit is in bytes
            (*info).vram_size = mem_limit / (1024 * 1024);
            AMDSMI_STATUS_SUCCESS
        }
        Err(error) => {
            tracing::error!("Failed to get pod memory usage: {error}");
            // Real values already written by the call above
            status
        }
    }
}

// --- dlsym-level interception ---

/// Intercept an SMI symbol at the dlsym level.
///
/// SMI libraries (rocm-smi, amd-smi) are loaded dynamically at runtime via
/// `dlopen`/`dlsym` (e.g., Python ctypes). Our LD_PRELOAD `dlsym` override
/// intercepts these lookups: we store the original function pointer and return
/// our detour's address so callers invoke our hook directly.
///
/// Returns `Some(detour_addr)` if we want to intercept this symbol,
/// or `None` to pass through the original.
pub(crate) unsafe fn try_intercept_smi_symbol(
    symbol_name: &str,
    original_addr: *const c_void,
) -> Option<*const c_void> {
    match symbol_name {
        "rsmi_dev_memory_total_get" => {
            // dlsym may be called multiple times for the same symbol, but the underlying
            // function pointer is the same regardless of handle. First-set wins; subsequent
            // sets log and are harmless no-ops.
            if FN_RSMI_DEV_MEMORY_TOTAL_GET
                .set(std::mem::transmute(original_addr))
                .is_err()
            {
                tracing::debug!("rsmi_dev_memory_total_get already intercepted, ignoring duplicate dlsym");
            }
            tracing::debug!("intercepting rsmi_dev_memory_total_get via dlsym");
            Some(rsmi_dev_memory_total_get_detour as *const c_void)
        }
        "amdsmi_get_gpu_memory_total" => {
            if FN_AMDSMI_GET_GPU_MEMORY_TOTAL
                .set(std::mem::transmute(original_addr))
                .is_err()
            {
                tracing::debug!("amdsmi_get_gpu_memory_total already intercepted, ignoring duplicate dlsym");
            }
            // Pre-resolve amdsmi_get_gpu_device_bdf from the same library.
            // Python ctypes loads libamd_smi with RTLD_LOCAL, making its symbols
            // invisible to RTLD_DEFAULT. We use dladdr on this known function pointer
            // to find the library and resolve the BDF function from it.
            // Only set on success — if resolution fails, leave the lock empty so the
            // other amdsmi symbol's arm can retry with its own original_addr.
            if let Some(f) = resolve_bdf_fn_from_lib(original_addr) {
                let _ = FN_AMDSMI_GET_BDF.set(Some(f));
            }
            tracing::debug!("intercepting amdsmi_get_gpu_memory_total via dlsym");
            Some(amdsmi_get_gpu_memory_total_detour as *const c_void)
        }
        "amdsmi_get_gpu_vram_info" => {
            if FN_AMDSMI_GET_GPU_VRAM_INFO
                .set(std::mem::transmute(original_addr))
                .is_err()
            {
                tracing::debug!("amdsmi_get_gpu_vram_info already intercepted, ignoring duplicate dlsym");
            }
            // Also try to resolve BDF function if not yet resolved
            if let Some(f) = resolve_bdf_fn_from_lib(original_addr) {
                let _ = FN_AMDSMI_GET_BDF.set(Some(f));
            }
            tracing::debug!("intercepting amdsmi_get_gpu_vram_info via dlsym");
            Some(amdsmi_get_gpu_vram_info_detour as *const c_void)
        }
        _ => None,
    }
}
