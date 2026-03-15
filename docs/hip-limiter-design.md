# hip-limiter Design

A cdylib loaded via `LD_PRELOAD` that intercepts HIP GPU memory allocation APIs, enforces per-pod VRAM limits, and reports usage to the hypervisor through shared memory.

## How It Works

```
Application (hipMalloc, hipFree, …)
        │
        ▼
LD_PRELOAD → hip-limiter.so (Frida GUM inline hooks)
        │
        ├─ Reserve-then-allocate (atomic fetch_add on SHM)
        ├─ Call real HIP API in libamdhip64.so
        ├─ Track allocation in DashMap
        └─ On free: remove from DashMap, saturating_fetch_sub on SHM
```

The limiter is transparent to applications. Frameworks see their pod's VRAM limit as the total GPU memory (via spoofed `hipMemGetInfo`/`hipDeviceTotalMem`/SMI queries), and allocations that exceed the limit return `hipErrorOutOfMemory`.

## Init Flow

1. **Library load** — `#[ctor] entry_point()` runs when `LD_PRELOAD` loads the cdylib.
2. **Config fetch** — blocking HTTP call to hypervisor API (`GET /api/v1/pod`) using K8s service account auth, with pod identity from `POD_NAME`, `POD_NAMESPACE`, `CONTAINER_NAME` env vars (injected via K8s downward API). Timeout: 15s connect, 30s total. If the hypervisor is unreachable, the limiter is never initialized and all hooks become passthrough (no enforcement). In test mode, reads `TF_SHM_FILE` and `TF_VISIBLE_DEVICES` env vars instead.
3. **Device mapping** — enumerates HIP devices, matches against config UUIDs by PCI BDF normalization (strips `amd-gpu-` prefix, lowercases).
4. **SHM attach** — deferred. SHM is lazily opened on first hook invocation via `OnceCell`, not at init time. If SHM is unavailable at that point, the hook falls through per-call rather than failing globally.
5. **Isolation check** — hooks activate when isolation mode is `"soft"` or unset (`None`). Only an explicitly non-`"soft"` value skips hook installation.
6. **Hook installation** — creates a Frida GUM `HookManager`, replaces 30 symbols in `libamdhip64.so` via inline hooks. Guarded by `catch_unwind` to prevent hook installation panics from crashing the host application. Installs a `dlsym` detour to catch late-loaded libraries (SMI libs loaded with `RTLD_LOCAL`). If `libamdhip64.so` is not yet loaded, installation is deferred until a `dlsym` call resolves a HIP symbol.

## Core Pattern: Reserve-Then-Allocate

Eliminates the TOCTOU race in check-then-allocate:

```
1. fetch_add(size) on pod_memory_used     → atomically reserve space
2. if new_total > limit → fetch_sub(size) → roll back, return OOM
3. call real hipMalloc (native allocator)
4. if native fails → fetch_sub(size)      → roll back reservation
5. on success → insert (pointer, size) into DashMap
```

The under-utilization window (between reserve and native call) is bounded by one allocation's duration. This is the accepted tradeoff for eliminating overcommit.

**Pitched allocations** (hipMallocPitch, hipMemAllocPitch, hipMalloc3D) use a two-phase variant: reserve an estimate (width × height), call native to learn actual pitch, then reserve the extra difference (pitch − width) × height. If the extra pushes over the limit, the initial reservation is rolled back and the native allocation is freed via the original (unhooked) `hipFree`.

**Free path** uses conservative ordering: call native free first, then decrement SHM via `saturating_fetch_sub`. A crash between free and decrement causes over-reporting (safe direction — prevents overcommit, never allows silent over-allocation).

## Components

### `hip_limiter.rs` — Entry Point

Globals: `GLOBAL_LIMITER: OnceLock<Limiter>` (the limiter instance), `HOOKS_INITIALIZED: AtomicBool` (whether hooks are installed), `GLOBAL_LIMITER_ERROR: OnceLock<String>` (records init failure), `LIMITER_ERROR_REPORTED: AtomicBool` (gates one-shot warning on first hooked call via CAS).

Also contains the `dlsym` detour with a `thread_local! IN_DLSYM_DETOUR` recursion guard to prevent infinite loops when Frida's own symbol resolution triggers `dlsym`.

### `limiter.rs` — Memory Accounting

The `Limiter` struct:

| Field | Type | Purpose |
|-------|------|---------|
| `shared_memory_handle` | `OnceCell<Arc<SharedMemoryHandle>>` | Lazily opened POSIX SHM with V2 device state |
| `gpu_idx_uuids` | `Vec<(usize, String)>` | Configured devices from hypervisor, sorted by index |
| `hip_device_mapping` | `DashMap<HipDevice, (usize, String)>` | Cache: HIP device ordinal → (raw_idx, uuid) |
| `allocation_tracker` | `DashMap<usize, (usize, u64)>` | pointer → (device_idx, size) |
| `isolation` | `Option<String>` | Must be `"soft"` or `None` for enforcement |

Key methods:
- `try_reserve(device_idx, size)` — atomic `fetch_add` on SHM, rollback if over limit
- `rollback_reservation(device_idx, size)` — undo a reservation after native failure
- `record_allocation(device_idx, ptr, size)` — insert into DashMap
- `record_free(ptr)` — remove from DashMap, `saturating_fetch_sub` on SHM
- `device_index_by_hip_device(hip_device)` — resolves HIP ordinal to SHM device index via PCI BDF
- `device_index_by_pci_bdf(bdf)` — resolves PCI BDF to device index (used by amdsmi hooks)

### `detour/mem.rs` — HIP API Hooks

30 Frida inline hooks on `libamdhip64.so` (19 alloc + 9 free + 2 info spoofing). Three macros drive the hook logic:

- **`check_and_alloc!`** — standard reserve-then-allocate for simple allocations (hipMalloc, hipHostMalloc, hipMallocAsync, arrays, mipmaps, etc.)
- **`check_and_alloc_pitched!`** — two-phase variant for pitched allocations where actual size depends on runtime pitch alignment
- **`check_and_free!`** — native free first, then decrement accounting

Each hook function uses the `with_device!` macro to resolve the current HIP device to a limiter device index before invoking these macros.

Also contains size computation helpers:
- `channel_desc_alloc_size` — `hipChannelFormatDesc` → bytes for flat arrays
- `channel_desc_bytes_per_elem` — extracts bytes-per-element from channel descriptor
- `array_format_bytes` — driver-API `hipArray_Format` → bytes per element
- `mip_chain_total_size` — iterative mip level summation with `MAX_MIP_LEVELS = 32` guard

### `detour/smi.rs` — SMI Spoofing

Hooks AMD System Management Interface libraries (`librocm_smi64.so`, `libamd_smi.so`) to report the pod's VRAM limit instead of physical GPU total. Uses `dlsym`-level interception (not inline Frida hooks) because these libraries are loaded with `RTLD_LOCAL`.

Hooked: `rsmi_dev_memory_total_get`, `amdsmi_get_gpu_memory_total`, `amdsmi_get_gpu_vram_info`.

### `config.rs` — Hypervisor Config

Fetches pod configuration from the Go hypervisor's REST API:
- Endpoint: `GET http://{HYPERVISOR_IP}:{HYPERVISOR_PORT}/api/v1/pod`
- Auth: K8s service account bearer token (from `/var/run/secrets/kubernetes.io/serviceaccount/token`)
- Returns: `gpu_uuids: Vec<String>` and `isolation: Option<String>` (device index resolution happens later in `Limiter::new()`)

### `hiplib.rs` — HIP FFI

Thin `libloading` wrapper around `libamdhip64.so` for non-hooked HIP queries (`hipGetDevice`, `hipGetDeviceCount`, `hipDeviceGetPCIBusId`).

## Shared Memory (SHM)

Binary layout shared between the Go hypervisor (writer) and Rust limiter (reader/writer):

```
SharedDeviceStateV2:
  devices: [DeviceEntryV2; 16]       — per-device state (max 16 devices)
    uuid: [u8; 64]                   — GPU UUID string
    is_active: AtomicU32             — device active flag
    device_info: SharedDeviceInfoV2
      mem_limit: AtomicU64           — VRAM limit in bytes (Go writes)
      pod_memory_used: AtomicU64     — current usage (Rust reads/writes)
      up_limit: AtomicU32            — utilization percentage
      total_cuda_cores: AtomicU32    — compute cores
      erl_*: ...                     — ERL token bucket fields (future compute enforcement)
  device_count: AtomicU32
  last_heartbeat: AtomicU64          — staleness detection (2s threshold)
  pids: ShmMutex<Set<usize, 2048>>   — tracked process IDs
```

**Go writes:** `mem_limit`, `up_limit`, `last_heartbeat`, device UUIDs.
**Rust writes:** `pod_memory_used` (via atomics — `fetch_add` on alloc, saturating CAS loop on free).

SHM is mounted as a shared `tmpfs` volume. `SHM_PATH` env var specifies the directory (default `/run/tensor-fusion/shm`); the actual file is `{SHM_PATH}/shm`. Must be accessible from both the hypervisor sidecar and application container.

## Allocation Tracker

`DashMap<usize, (usize, u64)>` — maps pointer address → (device_idx, allocation_size).

- Process-local: pointer addresses are virtual, only meaningful within one process
- The SHM `pod_memory_used` counter is the cross-process source of truth
- All pointer types share the same keyspace (device VA, host pointers, `hipArray_t`, `hipMipmappedArray_t`, `hipMemGenericAllocationHandle_t`) — their address spaces don't collide because each is a distinct heap allocation from the HIP runtime
- Zero-size allocations succeed without tracking (matches native HIP behavior)
- Free of unknown pointer is a no-op (handles zero-size ptrs and double-free gracefully)

## Atomics

**Reservation (alloc path):**
```rust
let previous = pod_memory_used.fetch_add(size, Ordering::AcqRel);
if previous.saturating_add(size) > limit {
    pod_memory_used.fetch_sub(size, Ordering::AcqRel);
    return Err(OverLimit);
}
```

`MAX_ALLOC_SIZE = u64::MAX / 2` guard rejects pathological sizes before `fetch_add` to prevent transient wrapping.

**Free path (saturating CAS loop):**

Implemented as `saturating_fetch_sub_pod_memory_used()` on `SharedDeviceInfoV2`:

```rust
loop {
    let current = pod_memory_used.load(Ordering::Acquire);
    let new_value = current.saturating_sub(size);
    match pod_memory_used.compare_exchange_weak(
        current, new_value, Ordering::AcqRel, Ordering::Acquire
    ) {
        Ok(_) => break,
        Err(_) => continue,  // retry on contention
    }
}
```

Saturating subtraction prevents underflow wrapping. Logs a warning if underflow would have occurred.

## Device Mapping

AMD GPU UUIDs are PCI BDF-based. Three naming conventions exist:
- HIP runtime: `0000:75:00.0` (bare BDF)
- Go hypervisor: `amd-gpu-0000:75:00.0` (lowercased)
- C provider: `AMD-GPU-0000:75:00.0` (uppercase prefix)

`normalize_uuid_to_bdf()` unifies all three by lowercasing and stripping the `amd-gpu-` prefix. Device resolution at runtime is cached in a `DashMap` to avoid repeated HIP API calls.

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `HYPERVISOR_IP` | Yes (production) | — | Hypervisor sidecar IP address |
| `HYPERVISOR_PORT` | Yes (production) | — | Hypervisor sidecar port |
| `POD_NAME` | Yes (production) | `""` | Pod name for hypervisor identification (K8s downward API) |
| `POD_NAMESPACE` | Yes (production) | `""` | Pod namespace for hypervisor identification |
| `CONTAINER_NAME` | Yes (production) | `""` | Container name for hypervisor identification |
| `SHM_PATH` | No | `/run/tensor-fusion/shm` | SHM directory (actual file is `{SHM_PATH}/shm`) |
| `ENABLE_HIP_HOOKS` | No | `true` | Set to `"false"` to disable all hooks (passthrough) |
| `TF_SKIP_HOOKS_IF_NO_LIMIT` | No | `false` | Skip hooks when all devices have `up_limit >= 100` |
| `TF_HIP_LIB_PATH` | No | `libamdhip64.so` | Override `libamdhip64.so` load path |
| `HTTP_REQUEST_TIMEOUT` | No | `30s` | Total HTTP request timeout (humantime format) |
| `HTTP_CONNECT_TIMEOUT` | No | `15s` | TCP connect timeout |
| `TF_ENABLE_LOG` | No | enabled | Set to `"off"`, `"0"`, or `"false"` to silence logs |
| `TF_LOG_PATH` | No | stdout | Path for rolling log file (daily rotation, 7 files) |
| `TF_LOG_LEVEL` | No | `INFO` | tracing `EnvFilter` directive |
| `TF_SHM_FILE` | No | — | Mock mode: use local file as SHM (bypasses hypervisor) |
| `TF_VISIBLE_DEVICES` | No | — | Mock mode: comma-separated GPU UUIDs |

## Failure Modes

| Scenario | Behavior |
|----------|----------|
| **Hypervisor unreachable at init** | Limiter is never initialized. All hooks become passthrough — no enforcement. Logged as warning on first hooked call. Init blocks application startup for up to 30s (connect + request timeout). |
| **SHM unavailable at runtime** | SHM is lazily opened on first allocation via `OnceCell::get_or_try_init`. If open fails, the hook falls through to the native call (passthrough). Subsequent calls retry the `OnceCell` init. |
| **SHM unavailable during free** | Pointer is removed from the DashMap tracker but `pod_memory_used` is never decremented, causing a permanent accounting leak for that allocation's size. |
| **Stale heartbeat** | Logged as warning but enforcement continues with last-known limits. Heartbeat threshold is 2 seconds. Can be noisy under load since it's checked on every `try_reserve` call. |
| **`libamdhip64.so` not loaded** | Hooks are deferred until a `dlsym` call resolves a HIP symbol. If the library is never loaded (non-GPU workload), the limiter is a silent no-op. |
| **Hook installation panic** | Caught by `catch_unwind`. Hooks are not installed, error is logged. Application continues without enforcement. |
| **Crash between free and decrement** | Over-reports memory usage (safe direction). Requires pod restart or SHM recreation to reset. |
| **`pod_memory_used` drift** | No manual reset mechanism. Hypervisor must recreate SHM or pod must be deleted. |

## Supporting Crates

| Crate | Purpose |
|-------|---------|
| **utils** | `HookManager` (Frida GUM wrapper), `SharedMemoryHandle` (POSIX SHM), V1/V2 SHM types, `HookFn<T>` (original function storage), `replace_symbol!` macro |
| **tf-macro** | `#[hook_fn]` proc macro — generates FFI type alias, static `HookFn`, and trace instrumentation for each detour function |
| **api-types** | `PodInfoResponse`, `PodInfo` — shared between Go hypervisor and Rust limiter |
| **hip-limiter-fuzz** | `SimulatedLimiter` / `MultiDeviceSimulatedLimiter` — pure-Rust model of memory accounting for proptest fuzzing (concurrent stress, edge cases, multi-device, pitched allocations) |

## Test Strategy

1. **Rust unit tests** (`cargo test -p hip-limiter -p hip-limiter-fuzz -p utils`) — size computation, UUID normalization, device resolution, SHM compat. No GPU needed.
2. **Proptest fuzzer** (`hip-limiter-fuzz`) — property-based tests for accounting invariants under concurrent access, edge cases (zero-size, max-size, double-free), multi-device routing.
3. **CTS on real GPU** (`tests/cts/`) — Python tests on MI325X via Docker with TheRock ROCm 7.11. Tests all hook variants, allocation enforcement, free tracking, info spoofing, concurrency, and edge cases.

See [hip-memory-hook-coverage.md](hip-memory-hook-coverage.md) for the full hook inventory and known gaps.
