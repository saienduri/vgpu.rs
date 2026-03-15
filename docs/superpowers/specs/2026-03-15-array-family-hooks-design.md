# Array Family Hooks — Design Spec

**Date**: 2026-03-15
**Scope**: Task 5 from the remaining HIP alloc hooks plan
**Status**: Approved

## Goal

Hook the 6 HIP array allocation/deallocation APIs so the hip-limiter accounts for array memory in the per-pod VRAM budget. Today, array allocations bypass enforcement entirely — this is the same gap that HAMi-core has (they compute sizes but never wire them into accounting).

## APIs to Hook

### 4 Allocation Hooks

| Symbol | Descriptor Type | Size Formula |
|--------|----------------|--------------|
| `hipMallocArray(array, desc, width, height, flags)` | `hipChannelFormatDesc` + explicit dims | `(x+y+z+w)/8 * width * max(height,1)` |
| `hipMalloc3DArray(array, desc, extent, flags)` | `hipChannelFormatDesc` + `hipExtent` | `(x+y+z+w)/8 * width * max(height,1) * max(depth,1)` |
| `hipArrayCreate(pHandle, pAllocateArray)` | `HIP_ARRAY_DESCRIPTOR` | `format_bytes * num_channels * width * max(height,1)` |
| `hipArray3DCreate(array, pAllocateArray)` | `HIP_ARRAY3D_DESCRIPTOR` | `format_bytes * num_channels * width * max(height,1) * max(depth,1)` |

### 2 Free Hooks

| Symbol | Signature |
|--------|-----------|
| `hipFreeArray(array)` | `array: hipArray_t` (`*mut c_void`) |
| `hipArrayDestroy(array)` | `array: hipArray_t` (`*mut c_void`) |

Both free a `hipArray_t`. They are separate symbols (runtime vs driver API) but identical in accounting behavior.

## Approach: Inline Size Computation + `check_and_alloc!`

Compute the allocation size from the descriptor before calling the existing `check_and_alloc!` macro. No new macros needed. The `hipArray_t` pointer serves as the DashMap tracking key (same approach as `hipMemCreate` handles — opaque pointers in disjoint keyspace from device VA pointers).

Free hooks use `check_and_free!` unchanged.

### Error policy

If size computation fails (unrecognized format enum, non-byte-aligned channel bits, overflow), the hook returns `hipErrorInvalidValue` (value `1`) and logs an error. **No silent passthrough** — unaccounted allocations are rejected to avoid silent budget leaks.

**Prerequisite**: Add `HIP_ERROR_INVALID_VALUE = 1` to `crates/hip-limiter/src/hiplib.rs` (currently only defines `HIP_ERROR_OUT_OF_MEMORY = 2` and `HIP_ERROR_UNKNOWN = 999`).

### Size accuracy

The naive formula undercounts slightly because the driver may apply internal tiling/padding for texture cache optimization. This is acceptable:
- Array allocs are rare in ML workloads (bulk memory is `hipMalloc`)
- Slight undercount favors the user (they get marginally more usable memory)
- No HIP API exists to query actual array memory consumption (`hipArrayGetMemoryRequirements` does not exist)
- This matches HAMi-core's intended approach (which they never finished wiring up)

## FFI Types

### Rust structs (in `detour/mem.rs`)

```rust
/// Runtime API channel format descriptor (hipChannelFormatDesc).
/// Fields x/y/z/w are bit widths per channel.
#[repr(C)]
struct HipChannelFormatDesc {
    x: c_int,
    y: c_int,
    z: c_int,
    w: c_int,
    f: c_int, // hipChannelFormatKind enum (signed int in C) — unused for size computation
}

/// Driver API 2D array descriptor (HIP_ARRAY_DESCRIPTOR).
#[repr(C)]
struct HipArrayDescriptor {
    width: usize,
    height: usize,
    format: c_int,  // hipArray_Format enum (signed int in C)
    num_channels: c_uint,
}

/// Driver API 3D array descriptor (HIP_ARRAY3D_DESCRIPTOR).
#[repr(C)]
struct HipArray3DDescriptor {
    width: usize,
    height: usize,
    depth: usize,
    format: c_int,  // hipArray_Format enum (signed int in C)
    num_channels: c_uint,
    flags: c_uint,
}
```

`HipExtent` already exists in `mem.rs` (used by `hipMalloc3D`).

**Note**: `hipMalloc3DArray` passes `hipExtent` **by value** (not by pointer), matching the existing `hipMalloc3D` hook pattern at `mem.rs:375`.

### Python ctypes structs (in `hip_helper.py`)

```python
class HipChannelFormatDesc(Structure):
    _fields_ = [("x", c_int), ("y", c_int), ("z", c_int), ("w", c_int), ("f", c_int)]

class HipArrayDescriptor(Structure):
    _fields_ = [("Width", c_size_t), ("Height", c_size_t),
                ("Format", c_int), ("NumChannels", c_uint)]

class HipArray3DDescriptor(Structure):
    _fields_ = [("Width", c_size_t), ("Height", c_size_t), ("Depth", c_size_t),
                ("Format", c_int), ("NumChannels", c_uint), ("Flags", c_uint)]
```

## Size Computation Helpers

### `array_format_bytes(format: c_int) -> Option<u64>`

Maps `hipArray_Format` enum to bytes per element:

| Enum Value | Constant | Bytes |
|-----------|----------|-------|
| 0x01 | `HIP_AD_FORMAT_UNSIGNED_INT8` | 1 |
| 0x02 | `HIP_AD_FORMAT_UNSIGNED_INT16` | 2 |
| 0x03 | `HIP_AD_FORMAT_UNSIGNED_INT32` | 4 |
| 0x08 | `HIP_AD_FORMAT_SIGNED_INT8` | 1 |
| 0x09 | `HIP_AD_FORMAT_SIGNED_INT16` | 2 |
| 0x0a | `HIP_AD_FORMAT_SIGNED_INT32` | 4 |
| 0x10 | `HIP_AD_FORMAT_HALF` | 2 |
| 0x20 | `HIP_AD_FORMAT_FLOAT` | 4 |
| other | — | `None` |

### `channel_desc_alloc_size(desc: &HipChannelFormatDesc, width: usize, height: usize) -> Option<u64>`

For runtime API arrays (`hipMallocArray`, `hipMalloc3DArray`):
1. `total_bits = x + y + z + w`
2. If `total_bits % 8 != 0` or `total_bits == 0` → `None`
3. `bytes_per_elem = total_bits / 8`
4. `bytes_per_elem * width * max(height, 1)` via checked arithmetic
5. Reject if result > `u64::MAX / 2`

For 3D: takes additional `depth` parameter, multiplies by `max(depth, 1)`.

### `driver_array_alloc_size(format: c_int, num_channels: c_uint, width: usize, height: usize) -> Option<u64>`

For driver API arrays (`hipArrayCreate`, `hipArray3DCreate`):
1. `elem_bytes = array_format_bytes(format)?`
2. `elem_bytes * num_channels * width * max(height, 1)` via checked arithmetic
3. Reject if result > `u64::MAX / 2`

For 3D: takes additional `depth` parameter, multiplies by `max(depth, 1)`.

## Hook Implementations

Each alloc hook: read descriptor → compute size → `check_and_alloc!`.
Each free hook: `check_and_free!` (unchanged).

### Registration order in `enable_hooks()`

Array free hooks registered before array alloc hooks (consistent with existing pattern — `check_and_alloc!` doesn't need array frees for rollback, but maintaining the convention keeps the code predictable). Placed after the existing pitched alloc hooks.

## Testing

### Unit tests in `mem.rs` (~15 tests)

**`array_format_bytes`**:
- All 8 valid format values → correct byte count
- Unknown values (0x00, 0x04, 0xFF) → `None`

**`channel_desc_alloc_size`** (runtime API):
- Typical RGBA float: x=32,y=32,z=32,w=32, 256x256 → 1 MiB
- Single-channel u8: x=8,y=0,z=0,w=0, 1024x512 → 512 KiB
- 1D (height=0): x=32, width=1000, height=0 → 4000
- Zero width → 0
- Non-byte-aligned (x=7) → `None`
- Overflow (width=usize::MAX) → `None`

**`driver_array_alloc_size`** (driver API):
- 2-channel FLOAT 512x512 → 2 MiB
- 3D 4-channel U8 64x64x64 → 1 MiB
- 1D (height=0) → treats as 1
- Unknown format → `None`
- Overflow → `None`

### CTS tests (Python, on real GPU)

**Allocation accounting** (`test_allocation_enforcement.py`):
- 4 new entries in `ALLOC_VARIANT_SCRIPTS` — one per alloc hook
- Each verifies SHM delta matches expected size

**OOM enforcement** (`test_allocation_enforcement.py`):
- 4 new entries in `VARIANT_SCRIPTS` using existing OOM template
- Set limit below array size, verify `hipErrorOutOfMemory`

**Free tracking** (`test_free_tracking.py`):
- `test_array_free_returns_to_zero` — `hipMallocArray` + `hipFreeArray`
- `test_array_destroy_returns_to_zero` — `hipArrayCreate` + `hipArrayDestroy`
- `test_array_cross_free` — `hipMallocArray` + `hipArrayDestroy` (cross-pairing)

**hip_helper.py additions**:
- 3 ctypes structs (listed above)
- 6 wrapper methods: `malloc_array`, `malloc_3d_array`, `array_create`, `array_3d_create`, `free_array`, `array_destroy`

### Fuzzer

No new fuzzer tests. The size computation helpers get unit tests. The reserve-then-allocate and free tracking invariants are already covered generically by the existing proptest fuzzer.

## Post-Task Hook Count

Current: 24 hooks (13 alloc + 5 free + 5 spoofing + 1 dlsym)
After Task 5: 30 hooks (+4 alloc + 2 free)
