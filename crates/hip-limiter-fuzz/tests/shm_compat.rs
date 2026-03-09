use std::sync::atomic::Ordering;

use utils::shared_memory::handle::SharedMemoryHandle;
use utils::shared_memory::{DeviceConfig, SharedDeviceState};

/// The SHM binary layout must be exactly 35504 bytes.
/// This is critical for cross-language compatibility with the Go hypervisor.
/// Size breakdown: enum header(8) + devices(16*136=2176) + counts(16) +
///   pids ShmMutex<Set>(32792) + padding(512)
#[test]
fn shm_file_size_matches_expected() {
    assert_eq!(
        std::mem::size_of::<SharedDeviceState>(),
        35504,
        "SharedDeviceState size must be exactly 35504 bytes for Go compatibility"
    );
}

/// The V2 discriminant must be at offset 0 in the enum layout.
/// Rust #[repr(C)] enums store the discriminant as the first field.
/// V1 = 0, V2 = 1.
#[test]
fn v2_discriminant_at_offset_zero() {
    let configs = vec![DeviceConfig {
        device_idx: 0,
        device_uuid: "test-gpu-0000:03:00.0".to_string(),
        up_limit: 80,
        mem_limit: 8 * 1024 * 1024 * 1024,
        sm_count: 82,
        max_thread_per_sm: 1536,
        total_cuda_cores: 2048,
    }];

    let state = SharedDeviceState::new(&configs);
    let state_ptr = &state as *const SharedDeviceState as *const u8;

    // Read the first 4 bytes as a u32 discriminant
    let discriminant = unsafe {
        let disc_ptr = state_ptr as *const u32;
        *disc_ptr
    };

    // SharedDeviceState::new() creates V2, so discriminant should be 1
    assert_eq!(
        discriminant, 1,
        "V2 discriminant must be 1 at offset 0"
    );
}

/// Create a SharedMemoryHandle with mock data, then read back and verify
/// that device configuration fields are correctly stored and retrievable.
#[test]
fn mock_shm_roundtrip() {
    let temp_dir = tempfile::TempDir::new().expect("failed to create temp dir");
    let shm_path = temp_dir.path().join("test_roundtrip");

    let configs = vec![
        DeviceConfig {
            device_idx: 0,
            device_uuid: "AMD-GPU-0000:03:00.0".to_string(),
            up_limit: 80,
            mem_limit: 8 * 1024 * 1024 * 1024,
            sm_count: 82,
            max_thread_per_sm: 1536,
            total_cuda_cores: 2048,
        },
        DeviceConfig {
            device_idx: 1,
            device_uuid: "AMD-GPU-0000:04:00.0".to_string(),
            up_limit: 90,
            mem_limit: 16 * 1024 * 1024 * 1024,
            sm_count: 110,
            max_thread_per_sm: 2048,
            total_cuda_cores: 4096,
        },
    ];

    let handle = SharedMemoryHandle::create(&shm_path, &configs)
        .expect("failed to create SHM");

    let state = handle.get_state();

    assert_eq!(state.device_count(), 2);
    assert_eq!(state.get_version(), 2);

    // Verify device 0
    let (uuid, _avail_cores, total_cores, mem_limit, pod_memory_used, up_limit, is_active) =
        state.get_device_info(0).expect("device 0 should exist");
    assert_eq!(uuid, "AMD-GPU-0000:03:00.0");
    assert_eq!(up_limit, 80);
    assert_eq!(mem_limit, 8 * 1024 * 1024 * 1024);
    assert_eq!(total_cores, 2048);
    assert_eq!(pod_memory_used, 0);
    assert!(is_active);

    // Verify device 1
    let (uuid, _, total_cores, mem_limit, _, up_limit, is_active) =
        state.get_device_info(1).expect("device 1 should exist");
    assert_eq!(uuid, "AMD-GPU-0000:04:00.0");
    assert_eq!(up_limit, 90);
    assert_eq!(mem_limit, 16 * 1024 * 1024 * 1024);
    assert_eq!(total_cores, 4096);
    assert!(is_active);
}

/// Verify that SharedMemoryHandle::create then ::open produces consistent reads.
#[test]
fn create_then_open_consistent() {
    let temp_dir = tempfile::TempDir::new().expect("failed to create temp dir");
    let shm_path = temp_dir.path().join("test_open_consistency");

    let configs = vec![DeviceConfig {
        device_idx: 0,
        device_uuid: "GPU-test-uuid-1234".to_string(),
        up_limit: 75,
        mem_limit: 4 * 1024 * 1024 * 1024,
        sm_count: 64,
        max_thread_per_sm: 1024,
        total_cuda_cores: 1024,
    }];

    let write_handle = SharedMemoryHandle::create(&shm_path, &configs)
        .expect("failed to create SHM");

    // Write some pod_memory_used
    write_handle
        .get_state()
        .set_pod_memory_used(0, 512 * 1024 * 1024);

    // Open from a separate handle and verify
    let read_handle = SharedMemoryHandle::open(&shm_path)
        .expect("failed to open SHM");

    let state = read_handle.get_state();
    assert_eq!(state.device_count(), 1);

    let (_, _, _, mem_limit, pod_memory_used, _, _) =
        state.get_device_info(0).expect("device 0 should exist");
    assert_eq!(mem_limit, 4 * 1024 * 1024 * 1024);
    assert_eq!(pod_memory_used, 512 * 1024 * 1024);
}

/// Verify heartbeat can be written and read back.
#[test]
fn heartbeat_roundtrip() {
    let temp_dir = tempfile::TempDir::new().expect("failed to create temp dir");
    let shm_path = temp_dir.path().join("test_heartbeat");

    let handle = SharedMemoryHandle::create(&shm_path, &[])
        .expect("failed to create SHM");

    let state = handle.get_state();
    let timestamp = 1709827200u64; // 2024-03-07 arbitrary timestamp
    state.update_heartbeat(timestamp);
    assert_eq!(state.get_last_heartbeat(), timestamp);
}

/// Verify that atomic fetch_add/fetch_sub on pod_memory_used works correctly
/// across two independent SharedMemoryHandle instances pointing at the same
/// backing SHM — the core correctness path for multi-process limiter accounting.
///
/// Gap: Existing tests either use a single handle (thread safety only) or
/// use set/get (non-atomic store). This exercises the actual fetch_add path
/// that try_reserve and record_free use in production.
#[test]
fn atomic_pod_memory_used_across_handles() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let shm_path = temp_dir.path().join("test_atomic_cross_handle");

    let configs = vec![DeviceConfig {
        device_idx: 0,
        device_uuid: "GPU-atomic-test".to_string(),
        up_limit: 80,
        mem_limit: 8 * 1024 * 1024 * 1024,
        sm_count: 82,
        max_thread_per_sm: 1536,
        total_cuda_cores: 2048,
    }];

    // Handle A creates the SHM (simulates hypervisor)
    let handle_a = SharedMemoryHandle::create(&shm_path, &configs).unwrap();
    // Handle B opens the same SHM (simulates limiter in another process)
    let handle_b = SharedMemoryHandle::open(&shm_path).unwrap();

    let state_a = handle_a.get_state();
    let state_b = handle_b.get_state();

    // fetch_add from handle A, read from handle B
    state_a.with_device(
        0,
        |d| d.device_info.pod_memory_used.fetch_add(1000, Ordering::AcqRel),
        |d| d.device_info.pod_memory_used.fetch_add(1000, Ordering::AcqRel),
    );
    let used_b = state_b
        .with_device(
            0,
            |d| d.device_info.get_pod_memory_used(),
            |d| d.device_info.get_pod_memory_used(),
        )
        .unwrap();
    assert_eq!(used_b, 1000);

    // fetch_add from handle B, read from handle A
    state_b.with_device(
        0,
        |d| d.device_info.pod_memory_used.fetch_add(500, Ordering::AcqRel),
        |d| d.device_info.pod_memory_used.fetch_add(500, Ordering::AcqRel),
    );
    let used_a = state_a
        .with_device(
            0,
            |d| d.device_info.get_pod_memory_used(),
            |d| d.device_info.get_pod_memory_used(),
        )
        .unwrap();
    assert_eq!(used_a, 1500);

    // fetch_sub from handle A (simulates record_free)
    state_a.with_device(
        0,
        |d| d.device_info.pod_memory_used.fetch_sub(1000, Ordering::AcqRel),
        |d| d.device_info.pod_memory_used.fetch_sub(1000, Ordering::AcqRel),
    );
    let used_b = state_b
        .with_device(
            0,
            |d| d.device_info.get_pod_memory_used(),
            |d| d.device_info.get_pod_memory_used(),
        )
        .unwrap();
    assert_eq!(used_b, 500);
}
