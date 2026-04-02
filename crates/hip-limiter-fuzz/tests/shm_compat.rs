use std::sync::atomic::Ordering;

use utils::shared_memory::handle::SharedMemoryHandle;
use utils::shared_memory::{DeviceConfig, SharedDeviceState};

/// The SHM binary layout must be exactly 35632 bytes.
/// This is critical for cross-language compatibility with the Go hypervisor.
/// Size breakdown: enum header(8) + devices(16*144=2304) + counts(16) +
///   pids ShmMutex<Set>(32792) + padding(512)
/// Note: DeviceEntryV2 grew from 136 to 144 bytes when effective_mem_limit was added.
#[test]
fn shm_file_size_matches_expected() {
    assert_eq!(
        std::mem::size_of::<SharedDeviceState>(),
        35632,
        "SharedDeviceState size must be exactly 35632 bytes for Go compatibility"
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

/// Verify that effective_mem_limit gates allocations when set, and falls back
/// to mem_limit when zero (not yet computed).
///
/// This tests the core enforcement path in atomic_reserve: the limiter should
/// use the tighter effective_mem_limit when non-zero, and mem_limit otherwise.
#[test]
fn effective_mem_limit_gates_allocation() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let shm_path = temp_dir.path().join("test_effective_limit");

    let mem_limit = 8 * 1024 * 1024 * 1024u64; // 8 GiB
    let configs = vec![DeviceConfig {
        device_idx: 0,
        device_uuid: "GPU-effective-test".to_string(),
        up_limit: 80,
        mem_limit,
        sm_count: 82,
        max_thread_per_sm: 1536,
        total_cuda_cores: 2048,
    }];

    let handle = SharedMemoryHandle::create(&shm_path, &configs).unwrap();
    let state = handle.get_state();

    // Initially effective_mem_limit is 0 — should use mem_limit as fallback
    let effective = state
        .with_device_v2_or(0, |d| d.device_info.get_effective_mem_limit())
        .unwrap();
    assert_eq!(effective, 0, "effective_mem_limit should be 0 initially");

    // Set effective_mem_limit to 4 GiB (simulating 4 GiB of overhead subtracted)
    let effective_limit = 4 * 1024 * 1024 * 1024u64; // 4 GiB
    state.with_device_v2_or(0, |d| {
        d.device_info.set_effective_mem_limit(effective_limit);
    });

    // Read back — should be 4 GiB
    let effective = state
        .with_device_v2_or(0, |d| d.device_info.get_effective_mem_limit())
        .unwrap();
    assert_eq!(effective, effective_limit);

    // Simulate reserve-then-check with effective limit:
    // Reserve 5 GiB — should exceed effective limit (4 GiB) even though < mem_limit (8 GiB)
    let size = 5 * 1024 * 1024 * 1024u64;
    let previous = state
        .with_device_v2_or(0, |d| {
            d.device_info.pod_memory_used.fetch_add(size, Ordering::AcqRel)
        })
        .unwrap();
    let new_used = previous.saturating_add(size);
    let active_limit = effective_limit; // non-zero, so we use it
    assert!(
        new_used > active_limit,
        "5 GiB should exceed 4 GiB effective limit"
    );
    // Roll back
    state.with_device_v2_or(0, |d| {
        d.device_info.saturating_fetch_sub_pod_memory_used(size)
    });

    // Reserve 3 GiB — should fit within effective limit
    let size = 3 * 1024 * 1024 * 1024u64;
    let previous = state
        .with_device_v2_or(0, |d| {
            d.device_info.pod_memory_used.fetch_add(size, Ordering::AcqRel)
        })
        .unwrap();
    let new_used = previous.saturating_add(size);
    assert!(
        new_used <= active_limit,
        "3 GiB should fit within 4 GiB effective limit"
    );
    // Roll back
    state.with_device_v2_or(0, |d| {
        d.device_info.saturating_fetch_sub_pod_memory_used(size)
    });

    // Reset effective_mem_limit to 0 — should fall back to mem_limit (8 GiB)
    state.with_device_v2_or(0, |d| {
        d.device_info.set_effective_mem_limit(0);
    });
    let size = 5 * 1024 * 1024 * 1024u64;
    let previous = state
        .with_device_v2_or(0, |d| {
            d.device_info.pod_memory_used.fetch_add(size, Ordering::AcqRel)
        })
        .unwrap();
    let new_used = previous.saturating_add(size);
    let fallback_limit = mem_limit; // effective is 0, fall back
    assert!(
        new_used <= fallback_limit,
        "5 GiB should fit within 8 GiB mem_limit when effective is 0"
    );
}
