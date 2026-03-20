//! KFD sysfs enumeration — discover GPU devices without touching the HIP runtime.
//!
//! This avoids initializing GPU context during standalone mode init, which would
//! break fork safety (child inherits stale GPU state).

use std::fmt;
use std::fs;
use std::path::Path;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A GPU device discovered via KFD sysfs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDevice {
    /// PCI Bus/Device/Function address, e.g. "0000:75:00.0"
    pub pci_bdf: String,
    /// DRM render node minor number, e.g. 128 → /dev/dri/renderD128
    pub render_minor: u32,
}

/// Parsed fields from a KFD topology node's `properties` file.
#[derive(Debug, Clone)]
pub struct NodeProperties {
    pub domain: u32,
    pub location_id: u32,
    pub drm_render_minor: u32,
}

/// Errors that can occur during GPU enumeration.
#[derive(Debug)]
pub enum EnumerationError {
    /// The KFD sysfs topology path does not exist or is inaccessible.
    SysfsNotAvailable(String),
    /// A required field is missing from a node's properties file.
    MissingField(String),
    /// A field value could not be parsed as an integer.
    ParseError(String),
    /// No GPU devices were found after scanning all nodes.
    NoDevicesFound,
}

impl fmt::Display for EnumerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SysfsNotAvailable(path) => write!(f, "KFD sysfs not available: {path}"),
            Self::MissingField(field) => write!(f, "missing required field: {field}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::NoDevicesFound => write!(f, "no GPU devices found"),
        }
    }
}

impl std::error::Error for EnumerationError {}

// ---------------------------------------------------------------------------
// BDF decoding
// ---------------------------------------------------------------------------

/// Decode a PCI Bus/Device/Function string from KFD domain and location_id.
///
/// KFD encodes BDF in `location_id` as:
/// - bits [15:8] → bus
/// - bits [7:3]  → device
/// - bits [2:0]  → function
pub fn decode_pci_bdf(domain: u32, location_id: u32) -> String {
    let bus = (location_id >> 8) & 0xFF;
    let device = (location_id >> 3) & 0x1F;
    let function = location_id & 0x7;
    format!("{domain:04x}:{bus:02x}:{device:02x}.{function}")
}

// ---------------------------------------------------------------------------
// Properties parsing
// ---------------------------------------------------------------------------

/// Parse required fields from the content of a KFD node `properties` file.
///
/// The file is line-oriented: `key value\n`. Values are decimal integers.
pub fn parse_properties(content: &str) -> Result<NodeProperties, EnumerationError> {
    let mut domain: Option<u32> = None;
    let mut location_id: Option<u32> = None;
    let mut drm_render_minor: Option<u32> = None;

    let parse_u32 = |key: &str, value: &str| -> Result<u32, EnumerationError> {
        value
            .parse()
            .map_err(|_| EnumerationError::ParseError(format!("{key}: {value}")))
    };

    for line in content.lines() {
        let mut parts = line.split_whitespace();
        let Some(key) = parts.next() else { continue };
        let Some(value) = parts.next() else { continue };

        match key {
            "domain" => domain = Some(parse_u32(key, value)?),
            "location_id" => location_id = Some(parse_u32(key, value)?),
            "drm_render_minor" => drm_render_minor = Some(parse_u32(key, value)?),
            _ => {}
        }
    }

    Ok(NodeProperties {
        domain: domain.ok_or_else(|| EnumerationError::MissingField("domain".into()))?,
        location_id: location_id
            .ok_or_else(|| EnumerationError::MissingField("location_id".into()))?,
        drm_render_minor: drm_render_minor
            .ok_or_else(|| EnumerationError::MissingField("drm_render_minor".into()))?,
    })
}

// ---------------------------------------------------------------------------
// Sysfs enumeration
// ---------------------------------------------------------------------------

const KFD_TOPOLOGY_PATH: &str = "/sys/class/kfd/kfd/topology/nodes";
const DRI_PATH: &str = "/dev/dri";

/// Enumerate GPU devices visible via KFD sysfs.
///
/// This is the public entry point. For tests, use [`enumerate_gpu_devices_from`].
pub fn enumerate_gpu_devices() -> Result<Vec<GpuDevice>, EnumerationError> {
    enumerate_gpu_devices_from(Path::new(KFD_TOPOLOGY_PATH), Path::new(DRI_PATH))
}

/// Testable enumeration function that accepts custom sysfs and dri paths.
pub fn enumerate_gpu_devices_from(
    topology_path: &Path,
    dri_path: &Path,
) -> Result<Vec<GpuDevice>, EnumerationError> {
    let entries = fs::read_dir(topology_path).map_err(|_| {
        EnumerationError::SysfsNotAvailable(topology_path.display().to_string())
    })?;

    // Collect node directory names and sort numerically.
    let mut node_ids: Vec<u32> = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name();
            name.to_str()?.parse::<u32>().ok()
        })
        .collect();
    node_ids.sort_unstable();

    let mut devices = Vec::new();

    for node_id in node_ids {
        let node_dir = topology_path.join(node_id.to_string());

        // Skip CPUs (gpu_id == 0) and inaccessible nodes (EPERM).
        let gpu_id_path = node_dir.join("gpu_id");
        let gpu_id_str = match fs::read_to_string(&gpu_id_path) {
            Ok(s) => s,
            Err(_) => continue, // EPERM or missing — skip
        };
        let gpu_id: u64 = match gpu_id_str.trim().parse() {
            Ok(id) => id,
            Err(_) => continue,
        };
        if gpu_id == 0 {
            continue; // CPU node
        }

        // Parse properties for this GPU node.
        let props_path = node_dir.join("properties");
        let props_content = match fs::read_to_string(&props_path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let props = match parse_properties(&props_content) {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!(node_id, "skipping KFD node: {e}");
                continue;
            }
        };

        // Check that the render device exists.
        let render_device = dri_path.join(format!("renderD{}", props.drm_render_minor));
        if !render_device.exists() {
            continue;
        }

        let pci_bdf = decode_pci_bdf(props.domain, props.location_id);
        devices.push(GpuDevice {
            pci_bdf,
            render_minor: props.drm_render_minor,
        });
    }

    if devices.is_empty() {
        return Err(EnumerationError::NoDevicesFound);
    }

    Ok(devices)
}

// ---------------------------------------------------------------------------
// Post-init verification
// ---------------------------------------------------------------------------

/// One-time verification: compare sysfs-enumerated BDFs against HIP runtime.
///
/// Called after HIP hooks are installed (meaning HIP library is loaded).
/// Uses hiplib's raw function pointers (not hooked) to query device info.
/// Logs an error on mismatch but doesn't change behavior.
pub(crate) fn verify_against_hip(sysfs_bdfs: &[String]) {
    let hip = crate::hiplib::hiplib();

    let device_count = match hip.get_device_count() {
        Ok(count) => count,
        Err(e) => {
            tracing::warn!("Post-init verification: hipGetDeviceCount failed: {e}");
            return;
        }
    };

    let mut hip_bdfs = Vec::with_capacity(device_count as usize);
    for i in 0..device_count {
        match hip.get_pci_bus_id(i) {
            Ok(bdf) => hip_bdfs.push(bdf.to_lowercase()),
            Err(e) => {
                tracing::warn!("Post-init verification: hipDeviceGetPCIBusId({i}) failed: {e}");
                return;
            }
        }
    }

    let sysfs_lower: Vec<String> = sysfs_bdfs.iter().map(|b| b.to_lowercase()).collect();

    if sysfs_lower.len() != hip_bdfs.len() {
        tracing::error!(
            sysfs_count = sysfs_lower.len(),
            hip_count = hip_bdfs.len(),
            sysfs_bdfs = ?sysfs_lower,
            hip_bdfs = ?hip_bdfs,
            "DEVICE MISMATCH: sysfs and HIP report different device counts"
        );
        return;
    }

    if sysfs_lower != hip_bdfs {
        tracing::error!(
            sysfs_bdfs = ?sysfs_lower,
            hip_bdfs = ?hip_bdfs,
            "DEVICE MISMATCH: sysfs and HIP report different device ordering"
        );
    } else {
        tracing::debug!(
            device_count,
            "Post-init verification: sysfs matches HIP device order"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- BDF decoding tests --

    #[test]
    fn test_decode_bdf_mi325x_device0() {
        assert_eq!(decode_pci_bdf(0, 0x7500), "0000:75:00.0");
    }

    #[test]
    fn test_decode_bdf_high_bus() {
        assert_eq!(decode_pci_bdf(0, 0xf500), "0000:f5:00.0");
    }

    #[test]
    fn test_decode_bdf_with_device_and_function() {
        // location_id=0x050b → bus=0x05, device=(0x0b >> 3)=0x01, function=0x0b & 0x7=0x3
        assert_eq!(decode_pci_bdf(0, 0x050b), "0000:05:01.3");
    }

    #[test]
    fn test_decode_bdf_nonzero_domain() {
        assert_eq!(decode_pci_bdf(1, 0x7500), "0001:75:00.0");
    }

    #[test]
    fn test_decode_bdf_zero_location() {
        assert_eq!(decode_pci_bdf(0, 0), "0000:00:00.0");
    }

    // -- Properties parsing tests --

    #[test]
    fn test_parse_properties_mi325x() {
        let content = "\
cpu_cores_count 0
simd_count 304
mem_banks_count 1
caches_count 228
io_links_count 3
p2p_links_count 0
max_waves_per_simd 16
lds_size_in_kb 64
gds_size_in_kb 0
num_gws 64
wave_front_size 64
array_count 38
simd_arrays_per_engine 2
cu_per_simd_array 2
simd_per_cu 4
max_slots_scratch_cu 32
gfx_target_version 120100
vendor_id 4098
device_id 29856
location_id 29952
domain 0
drm_render_minor 128
hive_id 11638818706288400384
num_sdma_engines 2
num_sdma_xgmi_engines 6
num_sdma_queues_per_engine 8
num_cp_queues 24
max_engine_clk_fcompute 2100
local_mem_size 206142726144
fw_version 262
capability 540676
sdma_fw_version 24
max_engine_clk_ccompute 2100
num_xcc 1
debug_prop 32768
unique_id 9895604649984";

        let props = parse_properties(content).unwrap();
        assert_eq!(props.domain, 0);
        assert_eq!(props.location_id, 29952);
        assert_eq!(props.drm_render_minor, 128);
    }

    #[test]
    fn test_parse_properties_missing_field() {
        let content = "domain 0\nlocation_id 29952\n";
        let result = parse_properties(content);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, EnumerationError::MissingField(ref f) if f == "drm_render_minor"),
            "expected MissingField(drm_render_minor), got: {err}"
        );
    }

    #[test]
    fn test_parse_properties_empty() {
        let result = parse_properties("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_properties_malformed_value() {
        let content = "domain abc\nlocation_id 29952\ndrm_render_minor 128\n";
        let result = parse_properties(content);
        assert!(matches!(result, Err(EnumerationError::ParseError(_))));
    }

    // -- Sysfs enumeration tests --

    /// Helper: create a mock KFD topology node directory.
    fn create_node(
        topology: &Path,
        node_id: u32,
        gpu_id: u64,
        properties: Option<&str>,
    ) {
        let node_dir = topology.join(node_id.to_string());
        fs::create_dir_all(&node_dir).unwrap();
        fs::write(node_dir.join("gpu_id"), gpu_id.to_string()).unwrap();
        if let Some(props) = properties {
            fs::write(node_dir.join("properties"), props).unwrap();
        }
    }

    fn gpu_properties(domain: u32, location_id: u32, render_minor: u32) -> String {
        format!(
            "domain {domain}\nlocation_id {location_id}\ndrm_render_minor {render_minor}\n"
        )
    }

    fn create_render_device(dri: &Path, minor: u32) {
        fs::write(dri.join(format!("renderD{minor}")), "").unwrap();
    }

    #[test]
    fn test_enumerate_single_gpu() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = tmp.path().join("nodes");
        let dri = tmp.path().join("dri");
        fs::create_dir_all(&topology).unwrap();
        fs::create_dir_all(&dri).unwrap();

        // Node 0: CPU
        create_node(&topology, 0, 0, None);
        // Node 1: GPU
        let props = gpu_properties(0, 0x7500, 128);
        create_node(&topology, 1, 12345, Some(&props));
        create_render_device(&dri, 128);

        let devices = enumerate_gpu_devices_from(&topology, &dri).unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].pci_bdf, "0000:75:00.0");
        assert_eq!(devices[0].render_minor, 128);
    }

    #[test]
    fn test_enumerate_multiple_gpus_sorted() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = tmp.path().join("nodes");
        let dri = tmp.path().join("dri");
        fs::create_dir_all(&topology).unwrap();
        fs::create_dir_all(&dri).unwrap();

        // Node 0: CPU
        create_node(&topology, 0, 0, None);
        // Node 2: GPU (created before node 1 to test sorting)
        let props2 = gpu_properties(0, 0xf500, 129);
        create_node(&topology, 2, 67890, Some(&props2));
        create_render_device(&dri, 129);
        // Node 1: GPU
        let props1 = gpu_properties(0, 0x7500, 128);
        create_node(&topology, 1, 12345, Some(&props1));
        create_render_device(&dri, 128);

        let devices = enumerate_gpu_devices_from(&topology, &dri).unwrap();
        assert_eq!(devices.len(), 2);
        // Should be sorted by node ID: node 1 first, node 2 second.
        assert_eq!(devices[0].pci_bdf, "0000:75:00.0");
        assert_eq!(devices[0].render_minor, 128);
        assert_eq!(devices[1].pci_bdf, "0000:f5:00.0");
        assert_eq!(devices[1].render_minor, 129);
    }

    #[test]
    fn test_enumerate_skips_missing_render_device() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = tmp.path().join("nodes");
        let dri = tmp.path().join("dri");
        fs::create_dir_all(&topology).unwrap();
        fs::create_dir_all(&dri).unwrap();

        create_node(&topology, 0, 0, None);

        // GPU 1 — has render device
        let props1 = gpu_properties(0, 0x7500, 128);
        create_node(&topology, 1, 12345, Some(&props1));
        create_render_device(&dri, 128);

        // GPU 2 — NO render device
        let props2 = gpu_properties(0, 0xf500, 129);
        create_node(&topology, 2, 67890, Some(&props2));

        let devices = enumerate_gpu_devices_from(&topology, &dri).unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].pci_bdf, "0000:75:00.0");
    }

    #[test]
    fn test_enumerate_no_gpus_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = tmp.path().join("nodes");
        let dri = tmp.path().join("dri");
        fs::create_dir_all(&topology).unwrap();
        fs::create_dir_all(&dri).unwrap();

        // Only a CPU node
        create_node(&topology, 0, 0, None);

        let result = enumerate_gpu_devices_from(&topology, &dri);
        assert!(matches!(result, Err(EnumerationError::NoDevicesFound)));
    }

    #[test]
    fn test_enumerate_skips_node_without_properties() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = tmp.path().join("nodes");
        let dri = tmp.path().join("dri");
        fs::create_dir_all(&topology).unwrap();
        fs::create_dir_all(&dri).unwrap();

        // Node 0: CPU
        create_node(&topology, 0, 0, None);
        // Node 1: GPU with gpu_id but no properties file
        create_node(&topology, 1, 12345, None);
        // Node 2: Valid GPU
        let props = gpu_properties(0, 0x7500, 128);
        create_node(&topology, 2, 67890, Some(&props));
        create_render_device(&dri, 128);

        let devices = enumerate_gpu_devices_from(&topology, &dri).unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].pci_bdf, "0000:75:00.0");
    }

    #[test]
    fn test_enumerate_sysfs_not_available() {
        let nonexistent = Path::new("/tmp/does-not-exist-kfd-test");
        let dri = Path::new("/tmp/does-not-exist-dri-test");

        let result = enumerate_gpu_devices_from(nonexistent, dri);
        assert!(matches!(result, Err(EnumerationError::SysfsNotAvailable(_))));
    }
}
