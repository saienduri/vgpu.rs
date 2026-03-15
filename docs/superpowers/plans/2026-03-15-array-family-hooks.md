# Array Family Hooks Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hook 6 HIP array allocation/deallocation APIs (hipMallocArray, hipMalloc3DArray, hipArrayCreate, hipArray3DCreate, hipFreeArray, hipArrayDestroy) so the hip-limiter accounts for array memory in the per-pod VRAM budget.

**Architecture:** Compute allocation size from descriptor structs (runtime: `hipChannelFormatDesc` bit widths; driver: `hipArray_Format` enum + `NumChannels`), then feed the size into the existing `check_and_alloc!` / `check_and_free!` macros. No new macros. Reject allocations with unrecognized formats (`hipErrorInvalidValue`) rather than allowing unaccounted passthrough.

**Tech Stack:** Rust (Frida GUM hooks), Python (CTS via ctypes), ROCm HIP API

**Spec:** `docs/superpowers/specs/2026-03-15-array-family-hooks-design.md`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `crates/hip-limiter/src/hiplib.rs` | Modify (line 12) | Add `HIP_ERROR_INVALID_VALUE` constant |
| `crates/hip-limiter/src/detour/mem.rs` | Modify (lines 1, 32, 397, 463, 671, 691) | Add FFI structs, size helpers, 6 hook functions, hook registrations, unit tests |
| `tests/cts/hip_helper.py` | Modify (lines 59, 232, 495) | Add ctypes structs, prototypes, wrapper methods |
| `tests/cts/test_allocation_enforcement.py` | Modify (lines 259, 799) | Add 4 alloc variants + 4 OOM variants |
| `tests/cts/test_free_tracking.py` | Modify (line 302) | Add 3 free tracking tests |

---

## Task 1: Add `HIP_ERROR_INVALID_VALUE` constant

**Files:**
- Modify: `crates/hip-limiter/src/hiplib.rs:11-12`
- Modify: `crates/hip-limiter/src/detour/mem.rs:7-8`

- [ ] **Step 1: Add the constant to hiplib.rs**

In `crates/hip-limiter/src/hiplib.rs`, after line 11 (`HIP_ERROR_OUT_OF_MEMORY`), add:

```rust
pub const HIP_ERROR_INVALID_VALUE: HipError = 1;
```

- [ ] **Step 2: Add the import to mem.rs**

In `crates/hip-limiter/src/detour/mem.rs`, update the import on lines 7-8 to include `HIP_ERROR_INVALID_VALUE`:

```rust
use crate::hiplib::{HipDevice, HipError, HipMemPool, HipStream, HIP_ERROR_INVALID_VALUE,
                    HIP_ERROR_OUT_OF_MEMORY, HIP_ERROR_UNKNOWN, HIP_SUCCESS};
```

Also add `c_int` to the `std::ffi` import on line 1:

```rust
use std::ffi::{c_int, c_uint, c_ulonglong, c_void};
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p hip-limiter 2>&1 | tail -5`
Expected: compiles with no errors (warnings about unused imports are fine at this stage)

- [ ] **Step 4: Commit**

```bash
git add crates/hip-limiter/src/hiplib.rs crates/hip-limiter/src/detour/mem.rs
git commit -m "feat(hip-limiter): add HIP_ERROR_INVALID_VALUE constant and c_int import"
```

---

## Task 2: Add FFI structs and size computation helpers with unit tests

**Files:**
- Modify: `crates/hip-limiter/src/detour/mem.rs:32` (after HipExtent), `196-203` (near checked_pitched_size), `691` (end of file for tests)

- [ ] **Step 1: Add FFI structs**

In `crates/hip-limiter/src/detour/mem.rs`, after line 32 (end of `HipExtent` struct), add:

```rust
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
```

- [ ] **Step 2: Add size computation helpers**

After the `checked_pitched_size` function (line 203), add three helpers:

```rust
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

/// Compute allocation size for runtime API array descriptors (hipChannelFormatDesc).
/// Returns `None` on non-byte-aligned bits, zero total bits, overflow, or size > u64::MAX/2.
fn channel_desc_alloc_size(
    desc: &HipChannelFormatDesc, width: usize, height: usize, depth: usize,
) -> Option<u64> {
    let total_bits = (desc.x as i64) + (desc.y as i64) + (desc.z as i64) + (desc.w as i64);
    if total_bits <= 0 || total_bits % 8 != 0 {
        return None;
    }
    let bytes_per_elem = total_bits as u64 / 8;
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
```

- [ ] **Step 3: Write unit tests**

At the end of `mem.rs` (after line 691, the closing `}` of `enable_hooks`), add:

```rust
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
}
```

- [ ] **Step 4: Run unit tests**

Run: `cargo test -p hip-limiter -- tests:: -v 2>&1 | tail -30`
Expected: all ~15 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/hip-limiter/src/detour/mem.rs
git commit -m "feat(hip-limiter): add array FFI structs, size computation helpers, and unit tests"
```

---

## Task 3: Add 6 hook functions and register them

**Files:**
- Modify: `crates/hip-limiter/src/detour/mem.rs` — after VMM hooks (~line 463), and in `enable_hooks()` (~line 671)

- [ ] **Step 1: Add 4 alloc hooks and 2 free hooks**

After the `hip_mem_release_detour` function (line 463) and before the `// --- Info spoofing hooks ---` comment (line 465), add:

```rust
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
```

- [ ] **Step 2: Register hooks in `enable_hooks()`**

After the `hipMalloc3D` registration (line 671) and before `hipMemGetInfo` (line 672), add the 6 new registrations. Array free hooks first, then alloc hooks:

```rust
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
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p hip-limiter 2>&1 | tail -5`
Expected: compiles with no errors

- [ ] **Step 4: Run all unit tests**

Run: `cargo test -p hip-limiter 2>&1 | tail -10`
Expected: all tests pass (unit tests + any existing tests)

- [ ] **Step 5: Commit**

```bash
git add crates/hip-limiter/src/detour/mem.rs
git commit -m "feat(hip-limiter): add hipMallocArray/hipMalloc3DArray/hipArrayCreate/hipArray3DCreate/hipFreeArray/hipArrayDestroy hooks"
```

---

## Task 4: Add Python ctypes structs, prototypes, and wrapper methods

**Files:**
- Modify: `tests/cts/hip_helper.py:59` (structs), `232` (prototypes), `495` (methods)

- [ ] **Step 1: Add ctypes struct definitions**

After the existing `HIPExtent` struct (line 99), add:

```python
# --- Array descriptor structs ---

# hipChannelFormatDesc — Runtime API channel format descriptor.
# Fields x/y/z/w are bit widths per channel.
HIP_AD_FORMAT_UNSIGNED_INT8 = 0x01
HIP_AD_FORMAT_FLOAT = 0x20

class HipChannelFormatDesc(Structure):
    _fields_ = [("x", c_int), ("y", c_int), ("z", c_int), ("w", c_int), ("f", c_int)]

class HipArrayDescriptor(Structure):
    _fields_ = [("Width", c_size_t), ("Height", c_size_t),
                ("Format", c_int), ("NumChannels", c_uint)]

class HipArray3DDescriptor(Structure):
    _fields_ = [("Width", c_size_t), ("Height", c_size_t), ("Depth", c_size_t),
                ("Format", c_int), ("NumChannels", c_uint), ("Flags", c_uint)]
```

- [ ] **Step 2: Add ctypes prototypes**

In the `_setup_prototypes` method, after the last prototype (`hipMalloc3D`, line 232), add:

```python
        # --- Array allocation prototypes ---
        self._lib.hipMallocArray.restype = c_int
        self._lib.hipMallocArray.argtypes = [POINTER(c_void_p), POINTER(HipChannelFormatDesc),
                                             c_size_t, c_size_t, c_uint]
        self._lib.hipMalloc3DArray.restype = c_int
        self._lib.hipMalloc3DArray.argtypes = [POINTER(c_void_p), POINTER(HipChannelFormatDesc),
                                               HIPExtent, c_uint]
        self._lib.hipArrayCreate.restype = c_int
        self._lib.hipArrayCreate.argtypes = [POINTER(c_void_p), POINTER(HipArrayDescriptor)]
        self._lib.hipArray3DCreate.restype = c_int
        self._lib.hipArray3DCreate.argtypes = [POINTER(c_void_p), POINTER(HipArray3DDescriptor)]
        self._lib.hipFreeArray.restype = c_int
        self._lib.hipFreeArray.argtypes = [c_void_p]
        self._lib.hipArrayDestroy.restype = c_int
        self._lib.hipArrayDestroy.argtypes = [c_void_p]
```

- [ ] **Step 3: Add wrapper methods**

After the last method in `HIPRuntime` (before the end of the class), add:

```python
    # --- Array allocation wrappers ---

    def malloc_array(self, width: int, height: int = 0,
                     desc_x: int = 32, desc_y: int = 0, desc_z: int = 0, desc_w: int = 0,
                     flags: int = 0) -> int:
        """Allocate a HIP array. Returns hipArray_t handle."""
        array = c_void_p(0)
        desc = HipChannelFormatDesc(x=desc_x, y=desc_y, z=desc_z, w=desc_w, f=2)  # f=2 = float
        self._check(
            self._lib.hipMallocArray(byref(array), byref(desc), width, height, flags),
            "hipMallocArray",
        )
        return array.value or 0

    def malloc_3d_array(self, width: int, height: int, depth: int,
                        desc_x: int = 32, desc_y: int = 0, desc_z: int = 0, desc_w: int = 0,
                        flags: int = 0) -> int:
        """Allocate a 3D HIP array. Returns hipArray_t handle."""
        array = c_void_p(0)
        desc = HipChannelFormatDesc(x=desc_x, y=desc_y, z=desc_z, w=desc_w, f=2)
        extent = HIPExtent(width=width, height=height, depth=depth)
        self._check(
            self._lib.hipMalloc3DArray(byref(array), byref(desc), extent, flags),
            "hipMalloc3DArray",
        )
        return array.value or 0

    def array_create(self, width: int, height: int = 0,
                     fmt: int = HIP_AD_FORMAT_FLOAT, num_channels: int = 1) -> int:
        """Create a HIP array via driver API. Returns hipArray_t handle."""
        array = c_void_p(0)
        desc = HipArrayDescriptor(Width=width, Height=height,
                                  Format=fmt, NumChannels=num_channels)
        self._check(
            self._lib.hipArrayCreate(byref(array), byref(desc)),
            "hipArrayCreate",
        )
        return array.value or 0

    def array_3d_create(self, width: int, height: int, depth: int,
                        fmt: int = HIP_AD_FORMAT_FLOAT, num_channels: int = 1,
                        flags: int = 0) -> int:
        """Create a 3D HIP array via driver API. Returns hipArray_t handle."""
        array = c_void_p(0)
        desc = HipArray3DDescriptor(Width=width, Height=height, Depth=depth,
                                    Format=fmt, NumChannels=num_channels, Flags=flags)
        self._check(
            self._lib.hipArray3DCreate(byref(array), byref(desc)),
            "hipArray3DCreate",
        )
        return array.value or 0

    def free_array(self, array: int) -> None:
        """Free a HIP array via hipFreeArray."""
        self._check(self._lib.hipFreeArray(c_void_p(array)), "hipFreeArray")

    def array_destroy(self, array: int) -> None:
        """Destroy a HIP array via hipArrayDestroy (driver API)."""
        self._check(self._lib.hipArrayDestroy(c_void_p(array)), "hipArrayDestroy")
```

- [ ] **Step 4: Verify Python syntax**

Run: `python3 -c "import ast; ast.parse(open('tests/cts/hip_helper.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add tests/cts/hip_helper.py
git commit -m "feat(cts): add array allocation ctypes structs and wrapper methods"
```

---

## Task 5: Add CTS allocation accounting and OOM tests

**Files:**
- Modify: `tests/cts/test_allocation_enforcement.py:259` (ALLOC_VARIANT_SCRIPTS), `799` (VARIANT_SCRIPTS)

- [ ] **Step 1: Add 4 entries to ALLOC_VARIANT_SCRIPTS**

After the `"hipMemCreate"` entry (line 258), before the closing `}` (line 259), add:

```python
    "hipMallocArray": """\
from hip_helper import HIPRuntime
hip = HIPRuntime()
arr = hip.malloc_array(width=256, height=256, desc_x=32)
print("ALLOC_OK")
hip.free_array(arr)
""",
    "hipMalloc3DArray": """\
from hip_helper import HIPRuntime
hip = HIPRuntime()
arr = hip.malloc_3d_array(width=64, height=64, depth=64, desc_x=32)
print("ALLOC_OK")
hip.free_array(arr)
""",
    "hipArrayCreate": """\
from hip_helper import HIPRuntime, HIP_AD_FORMAT_FLOAT
hip = HIPRuntime()
arr = hip.array_create(width=256, height=256, fmt=HIP_AD_FORMAT_FLOAT, num_channels=1)
print("ALLOC_OK")
hip.array_destroy(arr)
""",
    "hipArray3DCreate": """\
from hip_helper import HIPRuntime, HIP_AD_FORMAT_FLOAT
hip = HIPRuntime()
arr = hip.array_3d_create(width=64, height=64, depth=64, fmt=HIP_AD_FORMAT_FLOAT, num_channels=1)
print("ALLOC_OK")
hip.array_destroy(arr)
""",
```

- [ ] **Step 2: Add 4 entries to VARIANT_SCRIPTS (OOM)**

After the `"hipMemCreate"` entry (line 798), before the closing `}` (line 799), add:

```python
        "hipMallocArray": _oom_script(
            "arr = hip.malloc_array(width=over_size // 4, height=1, desc_x=32)",
            "hip.free_array(arr)"),
        "hipMalloc3DArray": _oom_script(
            "arr = hip.malloc_3d_array(width=over_size // 4, height=1, depth=1, desc_x=32)",
            "hip.free_array(arr)"),
        "hipArrayCreate": _oom_script(
            "arr = hip.array_create(width=over_size // 4, height=1)",
            "hip.array_destroy(arr)"),
        "hipArray3DCreate": _oom_script(
            "arr = hip.array_3d_create(width=over_size // 4, height=1, depth=1)",
            "hip.array_destroy(arr)"),
```

Note: `over_size // 4` because each element is 4 bytes (FLOAT format), so `elements * 4 = over_size` bytes.

- [ ] **Step 3: Verify Python syntax**

Run: `python3 -c "import ast; ast.parse(open('tests/cts/test_allocation_enforcement.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add tests/cts/test_allocation_enforcement.py
git commit -m "feat(cts): add array allocation accounting and OOM enforcement tests"
```

---

## Task 6: Add CTS free tracking tests

**Files:**
- Modify: `tests/cts/test_free_tracking.py:302` (end of file)

- [ ] **Step 1: Add 3 free tracking tests**

At the end of `test_free_tracking.py` (after line 301), add:

```python

# --- Array free tracking ---

def test_array_free_returns_to_zero(cts):
    """hipMallocArray + hipFreeArray should return pod_memory_used to zero."""
    result = cts.run_hip_test("""\
        import os
        from hip_helper import HIPRuntime
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]

        used_before = read_pod_memory_used(shm_path, 0)
        arr = hip.malloc_array(width=256, height=256, desc_x=32)
        used_after_alloc = read_pod_memory_used(shm_path, 0)
        hip.free_array(arr)
        used_after_free = read_pod_memory_used(shm_path, 0)

        expected = 4 * 256 * 256  # 32-bit float, 256x256

        print(f"delta_alloc={used_after_alloc - used_before}")
        print(f"expected={expected}")
        print(f"after_free={used_after_free}")

        if used_after_alloc - used_before == expected and used_after_free == used_before:
            print("PASS")
        else:
            print(f"FAIL: delta={used_after_alloc - used_before} expected={expected} after_free={used_after_free} before={used_before}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Array free tracking failed: {result.stdout}"


def test_array_destroy_returns_to_zero(cts):
    """hipArrayCreate + hipArrayDestroy should return pod_memory_used to zero."""
    result = cts.run_hip_test("""\
        import os
        from hip_helper import HIPRuntime, HIP_AD_FORMAT_FLOAT
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]

        used_before = read_pod_memory_used(shm_path, 0)
        arr = hip.array_create(width=256, height=256, fmt=HIP_AD_FORMAT_FLOAT, num_channels=1)
        used_after_alloc = read_pod_memory_used(shm_path, 0)
        hip.array_destroy(arr)
        used_after_free = read_pod_memory_used(shm_path, 0)

        expected = 4 * 1 * 256 * 256  # 4 bytes/elem * 1 channel * 256x256

        print(f"delta_alloc={used_after_alloc - used_before}")
        print(f"expected={expected}")
        print(f"after_free={used_after_free}")

        if used_after_alloc - used_before == expected and used_after_free == used_before:
            print("PASS")
        else:
            print(f"FAIL: delta={used_after_alloc - used_before} expected={expected} after_free={used_after_free} before={used_before}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Array destroy tracking failed: {result.stdout}"


def test_array_cross_free(cts):
    """hipMallocArray + hipArrayDestroy (cross-pairing) should return pod_memory_used to zero."""
    result = cts.run_hip_test("""\
        import os
        from hip_helper import HIPRuntime
        from shm_writer import read_pod_memory_used

        hip = HIPRuntime()
        shm_path = os.environ["TF_SHM_FILE"]

        used_before = read_pod_memory_used(shm_path, 0)
        arr = hip.malloc_array(width=256, height=256, desc_x=32)
        used_after_alloc = read_pod_memory_used(shm_path, 0)
        hip.array_destroy(arr)
        used_after_free = read_pod_memory_used(shm_path, 0)

        expected = 4 * 256 * 256

        print(f"delta_alloc={used_after_alloc - used_before}")
        print(f"expected={expected}")
        print(f"after_free={used_after_free}")

        if used_after_alloc - used_before == expected and used_after_free == used_before:
            print("PASS")
        else:
            print(f"FAIL: delta={used_after_alloc - used_before} expected={expected} after_free={used_after_free} before={used_before}")
    """)
    assert result.succeeded, f"Subprocess failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Array cross-free tracking failed: {result.stdout}"
```

- [ ] **Step 2: Verify Python syntax**

Run: `python3 -c "import ast; ast.parse(open('tests/cts/test_free_tracking.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tests/cts/test_free_tracking.py
git commit -m "feat(cts): add array free tracking tests (hipFreeArray, hipArrayDestroy, cross-pairing)"
```

---

## Task 7: Run full test loop

**Files:** None (validation only)

- [ ] **Step 1: Run Rust unit tests + fuzzer**

Run: `cd /home/sai/ws/super/projects/vgpu.rs && cargo test -p hip-limiter -p hip-limiter-fuzz -p utils 2>&1 | tail -20`
Expected: ~125+ tests pass (existing ~109 + ~15 new unit tests)

- [ ] **Step 2: Sync to remote and run CTS**

```bash
rsync -az --exclude target --exclude .git --exclude 'projects/' \
  -e "ssh -i ~/.ssh/vultr" \
  /home/sai/ws/super/projects/vgpu.rs/ z1_ossci@66.42.113.99:~/vgpu.rs/

ssh -i ~/.ssh/vultr z1_ossci@66.42.113.99 \
  'cd ~/vgpu.rs && bash scripts/run-cts.sh --cts-only'
```
Expected: ~55+ CTS tests pass (existing ~47 + 4 alloc variants + 4 OOM + 3 free tracking = ~58), 1 skipped (hipMallocFromPoolAsync)

- [ ] **Step 3: If any failures, debug and fix before proceeding**

Read test output carefully. Common issues:
- `hipErrorInvalidValue` in CTS → check ctypes struct layout matches C struct (field order, sizes)
- SHM delta mismatch → check size computation formula matches between Rust and Python test expectations
- `hipFreeArray` not found → check `libamdhip64.so` exports the symbol on the remote machine

---

## Task 8: Code review and final commit

- [ ] **Step 1: Run code review agent**

Dispatch a code review agent to review all changes against the design spec.

- [ ] **Step 2: Address review findings**

Fix any issues raised by the reviewer.

- [ ] **Step 3: Final commit (if review required changes)**

Commit any review-driven fixes.

- [ ] **Step 4: Push**

```bash
git push origin dev/saienduri/amd-support
```
