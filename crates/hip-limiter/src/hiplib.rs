use std::ffi::{c_char, c_void};

use once_cell::sync::OnceCell;

pub type HipError = i32;
pub type HipDevice = i32;
pub type HipStream = *mut c_void;
pub type HipMemPool = *mut c_void;

pub const HIP_SUCCESS: HipError = 0;
pub const HIP_ERROR_INVALID_VALUE: HipError = 1;
pub const HIP_ERROR_OUT_OF_MEMORY: HipError = 2;
pub const HIP_ERROR_UNKNOWN: HipError = 999;

const PRIMARY_HIP_LIB: &str = "libamdhip64.so";
const FALLBACK_HIP_LIB: &str = "libamdhip64.so.6";

type FnHipGetDevice = unsafe extern "C" fn(*mut HipDevice) -> HipError;
type FnHipGetDeviceCount = unsafe extern "C" fn(*mut i32) -> HipError;
type FnHipDeviceGetPCIBusId = unsafe extern "C" fn(*mut c_char, i32, HipDevice) -> HipError;

pub struct HipLib {
    _library: libloading::Library,
    pub hip_get_device: FnHipGetDevice,
    pub hip_get_device_count: FnHipGetDeviceCount,
    pub hip_device_get_pci_bus_id: FnHipDeviceGetPCIBusId,
}

// libloading::Library is Send+Sync when the loaded functions are thread-safe,
// which HIP runtime functions are.
unsafe impl Send for HipLib {}
unsafe impl Sync for HipLib {}

static HIP_LIB: OnceCell<HipLib> = OnceCell::new();

pub fn init_hiplib() -> Result<&'static HipLib, String> {
    HIP_LIB
        .get_or_try_init(|| HipLib::load().map_err(|e| format!("{e}")))
        .map_err(|e| format!("Failed to load HIP library: {e}"))
}

pub fn hiplib() -> &'static HipLib {
    HIP_LIB
        .get()
        .expect("HIP library not initialized; init_hiplib() must be called first")
}

impl HipLib {
    fn load() -> Result<Self, libloading::Error> {
        let mut candidates = Vec::with_capacity(3);
        if let Ok(path) = std::env::var("TF_HIP_LIB_PATH") {
            candidates.push(path);
        }
        candidates.push(PRIMARY_HIP_LIB.to_string());
        candidates.push(FALLBACK_HIP_LIB.to_string());

        let mut last_error: Option<libloading::Error> = None;

        for candidate in &candidates {
            tracing::info!("Loading HIP library from {candidate}");
            match unsafe { libloading::Library::new(candidate) } {
                Ok(library) => {
                    let hip_get_device: FnHipGetDevice = unsafe {
                        *library.get::<FnHipGetDevice>(b"hipGetDevice")?
                    };
                    let hip_get_device_count: FnHipGetDeviceCount = unsafe {
                        *library.get::<FnHipGetDeviceCount>(b"hipGetDeviceCount")?
                    };
                    let hip_device_get_pci_bus_id: FnHipDeviceGetPCIBusId = unsafe {
                        *library.get::<FnHipDeviceGetPCIBusId>(b"hipDeviceGetPCIBusId")?
                    };

                    tracing::info!("Successfully loaded HIP library from {candidate}");
                    return Ok(Self {
                        _library: library,
                        hip_get_device,
                        hip_get_device_count,
                        hip_device_get_pci_bus_id,
                    });
                }
                Err(error) => {
                    tracing::warn!(error = %error, "Failed to load {candidate}");
                    last_error = Some(error);
                }
            }
        }

        Err(last_error.expect("at least one candidate must have been tried"))
    }

    pub fn get_device_count(&self) -> Result<i32, HipError> {
        let mut count: i32 = 0;
        let result = unsafe { (self.hip_get_device_count)(&mut count) };
        if result == HIP_SUCCESS {
            Ok(count)
        } else {
            Err(result)
        }
    }

    pub fn get_pci_bus_id(&self, device: HipDevice) -> Result<String, HipError> {
        let mut buffer = [0u8; 64];
        let result = unsafe {
            (self.hip_device_get_pci_bus_id)(
                buffer.as_mut_ptr() as *mut c_char,
                buffer.len() as i32,
                device,
            )
        };
        if result != HIP_SUCCESS {
            return Err(result);
        }
        let nul_pos = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
        String::from_utf8(buffer[..nul_pos].to_vec())
            .map_err(|_| HIP_ERROR_UNKNOWN)
    }
}
