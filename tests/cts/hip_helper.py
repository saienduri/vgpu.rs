"""
HIP Helper — Thin ctypes wrapper around libamdhip64.so for direct HIP API calls.

Provides a HIPRuntime class that calls HIP runtime APIs directly via ctypes,
bypassing PyTorch's caching allocator. Works transparently whether or not
LD_PRELOAD with hip-limiter is active — the interposition is at the symbol level.

Usage:
    hip = HIPRuntime()                     # uses default /opt/rocm/lib/libamdhip64.so
    hip = HIPRuntime("/custom/path.so")    # custom library path

All methods raise HIPError on failure (non-zero hipError_t return).
"""

import ctypes
import os
from ctypes import POINTER, Structure, byref, c_char, c_int, c_size_t, c_ubyte, c_uint, c_ulonglong, c_ushort, c_void_p
from typing import Optional, Tuple


# ── HIP error codes ──

HIP_SUCCESS = 0
HIP_ERROR_INVALID_VALUE = 1
HIP_ERROR_OUT_OF_MEMORY = 2
HIP_ERROR_NOT_INITIALIZED = 3
HIP_ERROR_NOT_SUPPORTED = 801

# ── Virtual memory management types ──

# hipMemLocationType enum
HIP_MEM_LOCATION_TYPE_DEVICE = 1

# hipMemAllocationType enum
HIP_MEM_ALLOCATION_TYPE_PINNED = 0x1

# hipMemAllocationGranularity_flags enum
HIP_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0


class HipMemLocation(Structure):
    _fields_ = [("type", c_int), ("id", c_int)]


class _HipMemAllocFlags(Structure):
    _fields_ = [
        ("compressionType", c_ubyte),
        ("gpuDirectRDMACapable", c_ubyte),
        ("usage", c_ushort),
    ]


class HipMemAllocationProp(Structure):
    _fields_ = [
        ("type", c_int),                    # hipMemAllocationType
        ("requestedHandleType", c_int),     # hipMemAllocationHandleType (union)
        ("location", HipMemLocation),
        ("win32HandleMetaData", c_void_p),
        ("allocFlags", _HipMemAllocFlags),
    ]


class HIPError(Exception):
    """Exception raised when a HIP API call returns a non-zero error code."""

    ERROR_NAMES = {
        0: "hipSuccess",
        1: "hipErrorInvalidValue",
        2: "hipErrorOutOfMemory",
        3: "hipErrorNotInitialized",
        100: "hipErrorNoDevice",
        101: "hipErrorInvalidDevice",
        999: "hipErrorUnknown",
    }

    def __init__(self, error_code: int, api_name: str = ""):
        self.error_code = error_code
        self.api_name = api_name
        name = self.ERROR_NAMES.get(error_code, f"hipError({error_code})")
        msg = f"{api_name}: {name} (code {error_code})" if api_name else f"{name} (code {error_code})"
        super().__init__(msg)


class HIPPitchedPtr(ctypes.Structure):
    """ctypes mirror of hipPitchedPtr (used by hipMalloc3D)."""
    _fields_ = [
        ("ptr", c_void_p),
        ("pitch", c_size_t),
        ("xsize", c_size_t),
        ("ysize", c_size_t),
    ]


class HIPExtent(ctypes.Structure):
    """ctypes mirror of hipExtent (used by hipMalloc3D)."""
    _fields_ = [
        ("width", c_size_t),
        ("height", c_size_t),
        ("depth", c_size_t),
    ]


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


def _default_hip_lib_path() -> str:
    """Determine the default path to libamdhip64.so."""
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    return os.path.join(rocm_path, "lib", "libamdhip64.so")


class HIPRuntime:
    """Thin ctypes wrapper around the HIP runtime library.

    All allocation/free methods return raw integer pointers (device or host).
    Error checking is done via hipError_t return codes.
    """

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = _default_hip_lib_path()

        self._lib = ctypes.CDLL(lib_path)
        self._setup_prototypes()

    def _setup_prototypes(self):
        """Set up ctypes function prototypes for all HIP APIs we use."""
        lib = self._lib

        # hipError_t hipGetDeviceCount(int* count)
        lib.hipGetDeviceCount.restype = c_int
        lib.hipGetDeviceCount.argtypes = [POINTER(c_int)]

        # hipError_t hipSetDevice(int deviceId)
        lib.hipSetDevice.restype = c_int
        lib.hipSetDevice.argtypes = [c_int]

        # hipError_t hipGetDevice(int* deviceId)
        lib.hipGetDevice.restype = c_int
        lib.hipGetDevice.argtypes = [POINTER(c_int)]

        # hipError_t hipMalloc(void** ptr, size_t size)
        lib.hipMalloc.restype = c_int
        lib.hipMalloc.argtypes = [POINTER(c_void_p), c_size_t]

        # hipError_t hipFree(void* ptr)
        lib.hipFree.restype = c_int
        lib.hipFree.argtypes = [c_void_p]

        # hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags)
        lib.hipHostMalloc.restype = c_int
        lib.hipHostMalloc.argtypes = [POINTER(c_void_p), c_size_t, c_uint]

        # hipError_t hipHostFree(void* ptr)
        lib.hipHostFree.restype = c_int
        lib.hipHostFree.argtypes = [c_void_p]

        # hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream)
        lib.hipMallocAsync.restype = c_int
        lib.hipMallocAsync.argtypes = [POINTER(c_void_p), c_size_t, c_void_p]

        # hipError_t hipFreeAsync(void* dev_ptr, hipStream_t stream)
        lib.hipFreeAsync.restype = c_int
        lib.hipFreeAsync.argtypes = [c_void_p, c_void_p]

        # hipError_t hipMemGetInfo(size_t* free, size_t* total)
        lib.hipMemGetInfo.restype = c_int
        lib.hipMemGetInfo.argtypes = [POINTER(c_size_t), POINTER(c_size_t)]

        # hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device)
        lib.hipDeviceTotalMem.restype = c_int
        lib.hipDeviceTotalMem.argtypes = [POINTER(c_size_t), c_int]

        # hipError_t hipMallocManaged(void** dev_ptr, size_t size, unsigned int flags)
        lib.hipMallocManaged.restype = c_int
        lib.hipMallocManaged.argtypes = [POINTER(c_void_p), c_size_t, c_uint]

        # hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags)
        lib.hipExtMallocWithFlags.restype = c_int
        lib.hipExtMallocWithFlags.argtypes = [POINTER(c_void_p), c_size_t, c_uint]

        # hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height)
        lib.hipMallocPitch.restype = c_int
        lib.hipMallocPitch.argtypes = [POINTER(c_void_p), POINTER(c_size_t), c_size_t, c_size_t]

        # hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height, unsigned int elementSizeBytes)
        lib.hipMemAllocPitch.restype = c_int
        lib.hipMemAllocPitch.argtypes = [POINTER(c_void_p), POINTER(c_size_t), c_size_t, c_size_t, c_uint]

        # hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size, const hipMemAllocationProp* prop, unsigned long long flags)
        lib.hipMemCreate.restype = c_int
        lib.hipMemCreate.argtypes = [POINTER(c_void_p), c_size_t, POINTER(HipMemAllocationProp), c_ulonglong]

        # hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle)
        lib.hipMemRelease.restype = c_int
        lib.hipMemRelease.argtypes = [c_void_p]

        # hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop, hipMemAllocationGranularity_flags option)
        lib.hipMemGetAllocationGranularity.restype = c_int
        lib.hipMemGetAllocationGranularity.argtypes = [POINTER(c_size_t), POINTER(HipMemAllocationProp), c_uint]

        # hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device)
        lib.hipDeviceGetPCIBusId.restype = c_int
        lib.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, c_int, c_int]

        # hipError_t hipDeviceSynchronize()
        lib.hipDeviceSynchronize.restype = c_int
        lib.hipDeviceSynchronize.argtypes = []

        # hipError_t hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream)
        lib.hipMallocFromPoolAsync.restype = c_int
        lib.hipMallocFromPoolAsync.argtypes = [POINTER(c_void_p), c_size_t, c_void_p, c_void_p]

        # hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device)
        lib.hipDeviceGetDefaultMemPool.restype = c_int
        lib.hipDeviceGetDefaultMemPool.argtypes = [POINTER(c_void_p), c_int]

        # hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags)
        lib.hipHostAlloc.restype = c_int
        lib.hipHostAlloc.argtypes = [POINTER(c_void_p), c_size_t, c_uint]

        # hipError_t hipMallocHost(void** ptr, size_t size)
        lib.hipMallocHost.restype = c_int
        lib.hipMallocHost.argtypes = [POINTER(c_void_p), c_size_t]

        # hipError_t hipMemAllocHost(void** ptr, size_t size)
        lib.hipMemAllocHost.restype = c_int
        lib.hipMemAllocHost.argtypes = [POINTER(c_void_p), c_size_t]

        # hipError_t hipFreeHost(void* ptr)
        lib.hipFreeHost.restype = c_int
        lib.hipFreeHost.argtypes = [c_void_p]

        # hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent)
        lib.hipMalloc3D.restype = c_int
        lib.hipMalloc3D.argtypes = [POINTER(HIPPitchedPtr), HIPExtent]

        # --- Array allocation prototypes ---
        lib.hipMallocArray.restype = c_int
        lib.hipMallocArray.argtypes = [POINTER(c_void_p), POINTER(HipChannelFormatDesc),
                                       c_size_t, c_size_t, c_uint]
        lib.hipMalloc3DArray.restype = c_int
        lib.hipMalloc3DArray.argtypes = [POINTER(c_void_p), POINTER(HipChannelFormatDesc),
                                         HIPExtent, c_uint]
        lib.hipArrayCreate.restype = c_int
        lib.hipArrayCreate.argtypes = [POINTER(c_void_p), POINTER(HipArrayDescriptor)]
        lib.hipArray3DCreate.restype = c_int
        lib.hipArray3DCreate.argtypes = [POINTER(c_void_p), POINTER(HipArray3DDescriptor)]
        lib.hipFreeArray.restype = c_int
        lib.hipFreeArray.argtypes = [c_void_p]
        lib.hipArrayDestroy.restype = c_int
        lib.hipArrayDestroy.argtypes = [c_void_p]
        lib.hipMallocMipmappedArray.restype = c_int
        lib.hipMallocMipmappedArray.argtypes = [POINTER(c_void_p), POINTER(HipChannelFormatDesc),
                                                 HIPExtent, c_uint, c_uint]
        lib.hipMipmappedArrayCreate.restype = c_int
        lib.hipMipmappedArrayCreate.argtypes = [POINTER(c_void_p), POINTER(HipArray3DDescriptor),
                                                 c_uint]
        lib.hipFreeMipmappedArray.restype = c_int
        lib.hipFreeMipmappedArray.argtypes = [c_void_p]
        lib.hipMipmappedArrayDestroy.restype = c_int
        lib.hipMipmappedArrayDestroy.argtypes = [c_void_p]

    def _check(self, result: int, api_name: str) -> int:
        """Check a HIP API return code and raise HIPError if non-zero."""
        if result != HIP_SUCCESS:
            raise HIPError(result, api_name)
        return result

    # ── Device management ──

    def get_device_count(self) -> int:
        """Return the number of HIP-capable devices."""
        count = c_int(0)
        self._check(self._lib.hipGetDeviceCount(byref(count)), "hipGetDeviceCount")
        return count.value

    def set_device(self, device_id: int) -> None:
        """Set the current device for the calling thread."""
        self._check(self._lib.hipSetDevice(device_id), "hipSetDevice")

    def get_device(self) -> int:
        """Get the current device for the calling thread."""
        device_id = c_int(0)
        self._check(self._lib.hipGetDevice(byref(device_id)), "hipGetDevice")
        return device_id.value

    def get_pci_bus_id(self, device: int) -> str:
        """Get the PCI bus ID string for a device."""
        buf = ctypes.create_string_buffer(64)
        self._check(
            self._lib.hipDeviceGetPCIBusId(buf, 64, device),
            "hipDeviceGetPCIBusId",
        )
        return buf.value.decode("utf-8")

    def device_synchronize(self) -> None:
        """Synchronize the current device."""
        self._check(self._lib.hipDeviceSynchronize(), "hipDeviceSynchronize")

    # ── Memory allocation ──

    def malloc(self, size: int) -> int:
        """Allocate device memory. Returns device pointer as integer."""
        ptr = c_void_p(0)
        self._check(self._lib.hipMalloc(byref(ptr), size), "hipMalloc")
        return ptr.value or 0

    def malloc_raw(self, size: int) -> int:
        """Like malloc but returns the raw hipError_t code instead of raising."""
        ptr = c_void_p(0)
        return self._lib.hipMalloc(byref(ptr), size), (ptr.value or 0)

    def free(self, ptr: int) -> None:
        """Free device memory."""
        self._check(self._lib.hipFree(c_void_p(ptr)), "hipFree")

    def free_raw(self, ptr: int) -> int:
        """Like free but returns the raw hipError_t code instead of raising."""
        return self._lib.hipFree(c_void_p(ptr))

    def host_malloc(self, size: int, flags: int = 0) -> int:
        """Allocate pinned host memory. Returns host pointer as integer."""
        ptr = c_void_p(0)
        self._check(
            self._lib.hipHostMalloc(byref(ptr), size, flags),
            "hipHostMalloc",
        )
        return ptr.value or 0

    def host_free(self, ptr: int) -> None:
        """Free pinned host memory."""
        self._check(self._lib.hipHostFree(c_void_p(ptr)), "hipHostFree")

    def malloc_async(self, size: int, stream: int = 0) -> int:
        """Allocate device memory asynchronously. Returns device pointer."""
        ptr = c_void_p(0)
        self._check(
            self._lib.hipMallocAsync(byref(ptr), size, c_void_p(stream)),
            "hipMallocAsync",
        )
        return ptr.value or 0

    def free_async(self, ptr: int, stream: int = 0) -> None:
        """Free device memory asynchronously."""
        self._check(
            self._lib.hipFreeAsync(c_void_p(ptr), c_void_p(stream)),
            "hipFreeAsync",
        )

    def malloc_managed(self, size: int, flags: int = 1) -> int:
        """Allocate managed memory. Returns pointer."""
        ptr = c_void_p(0)
        self._check(
            self._lib.hipMallocManaged(byref(ptr), size, flags),
            "hipMallocManaged",
        )
        return ptr.value or 0

    def ext_malloc_with_flags(self, size: int, flags: int = 0) -> int:
        """Allocate device memory with flags (AMD extension). Returns pointer."""
        ptr = c_void_p(0)
        self._check(
            self._lib.hipExtMallocWithFlags(byref(ptr), size, flags),
            "hipExtMallocWithFlags",
        )
        return ptr.value or 0

    def malloc_pitch(self, width: int, height: int) -> Tuple[int, int]:
        """Allocate pitched device memory. Returns (pointer, pitch)."""
        ptr = c_void_p(0)
        pitch = c_size_t(0)
        self._check(
            self._lib.hipMallocPitch(byref(ptr), byref(pitch), width, height),
            "hipMallocPitch",
        )
        return (ptr.value or 0), pitch.value

    def mem_alloc_pitch(self, width: int, height: int, element_size: int = 4) -> Tuple[int, int]:
        """Allocate pitched device memory via hipMemAllocPitch (driver API). Returns (pointer, pitch)."""
        ptr = c_void_p(0)
        pitch = c_size_t(0)
        self._check(
            self._lib.hipMemAllocPitch(byref(ptr), byref(pitch), width, height, element_size),
            "hipMemAllocPitch",
        )
        return (ptr.value or 0), pitch.value

    def host_alloc(self, size: int, flags: int = 0) -> int:
        """Allocate pinned host memory via hipHostAlloc (alias of hipHostMalloc). Returns host pointer."""
        ptr = c_void_p(0)
        self._check(
            self._lib.hipHostAlloc(byref(ptr), size, flags),
            "hipHostAlloc",
        )
        return ptr.value or 0

    def malloc_host(self, size: int) -> int:
        """Allocate pinned host memory via hipMallocHost (deprecated, no flags). Returns host pointer."""
        ptr = c_void_p(0)
        self._check(
            self._lib.hipMallocHost(byref(ptr), size),
            "hipMallocHost",
        )
        return ptr.value or 0

    def mem_alloc_host(self, size: int) -> int:
        """Allocate pinned host memory via hipMemAllocHost (deprecated, no flags). Returns host pointer."""
        ptr = c_void_p(0)
        self._check(
            self._lib.hipMemAllocHost(byref(ptr), size),
            "hipMemAllocHost",
        )
        return ptr.value or 0

    def free_host(self, ptr: int) -> None:
        """Free pinned host memory via hipFreeHost (alias of hipHostFree)."""
        self._check(self._lib.hipFreeHost(c_void_p(ptr)), "hipFreeHost")

    def free_host_raw(self, ptr: int) -> int:
        """Like free_host but returns the raw hipError_t code instead of raising."""
        return self._lib.hipFreeHost(c_void_p(ptr))

    def get_allocation_granularity(self, device: int = 0) -> int:
        """Get the minimum allocation granularity for hipMemCreate on a device."""
        granularity = c_size_t(0)
        prop = HipMemAllocationProp(
            type=HIP_MEM_ALLOCATION_TYPE_PINNED,
            location=HipMemLocation(type=HIP_MEM_LOCATION_TYPE_DEVICE, id=device),
        )
        self._check(
            self._lib.hipMemGetAllocationGranularity(
                byref(granularity), byref(prop), HIP_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
            "hipMemGetAllocationGranularity",
        )
        return granularity.value

    def mem_create(self, size: int, device: int = 0) -> int:
        """Allocate physical GPU memory via hipMemCreate. Returns opaque handle as integer.

        size must be aligned to the allocation granularity (use get_allocation_granularity()).
        """
        handle = c_void_p(0)
        prop = HipMemAllocationProp(
            type=HIP_MEM_ALLOCATION_TYPE_PINNED,
            location=HipMemLocation(type=HIP_MEM_LOCATION_TYPE_DEVICE, id=device),
        )
        self._check(
            self._lib.hipMemCreate(byref(handle), size, byref(prop), 0),
            "hipMemCreate",
        )
        return handle.value or 0

    def mem_create_raw(self, size: int, device: int = 0) -> Tuple[int, int]:
        """Like mem_create but returns (error_code, handle) instead of raising."""
        handle = c_void_p(0)
        prop = HipMemAllocationProp(
            type=HIP_MEM_ALLOCATION_TYPE_PINNED,
            location=HipMemLocation(type=HIP_MEM_LOCATION_TYPE_DEVICE, id=device),
        )
        err = self._lib.hipMemCreate(byref(handle), size, byref(prop), 0)
        return err, (handle.value or 0)

    def mem_release(self, handle: int) -> None:
        """Release physical GPU memory via hipMemRelease."""
        self._check(self._lib.hipMemRelease(c_void_p(handle)), "hipMemRelease")

    def malloc_3d(self, width: int, height: int, depth: int) -> Tuple[int, int, int, int]:
        """Allocate 3D pitched device memory via hipMalloc3D.

        Returns (pointer, pitch, xsize, ysize).
        """
        pitched = HIPPitchedPtr()
        extent = HIPExtent(width=width, height=height, depth=depth)
        self._check(
            self._lib.hipMalloc3D(byref(pitched), extent),
            "hipMalloc3D",
        )
        return (pitched.ptr or 0), pitched.pitch, pitched.xsize, pitched.ysize

    def get_default_mem_pool(self, device: int = 0):
        """Get the default memory pool for a device. Returns opaque pool handle (c_void_p)."""
        mem_pool = c_void_p(0)
        self._check(
            self._lib.hipDeviceGetDefaultMemPool(byref(mem_pool), device),
            "hipDeviceGetDefaultMemPool",
        )
        return mem_pool

    def malloc_from_pool_async(self, size: int, mem_pool, stream=None) -> int:
        """hipMallocFromPoolAsync(devPtr, size, memPool, stream).

        Requires a valid memory pool handle obtained from hipDeviceGetDefaultMemPool
        or hipMemPoolCreate.
        """
        ptr = c_void_p(0)
        if stream is None:
            stream = c_void_p(0)
        err = self._lib.hipMallocFromPoolAsync(
            byref(ptr), c_size_t(size), mem_pool, stream
        )
        self._check(err, "hipMallocFromPoolAsync")
        return ptr.value or 0

    # ── Memory info ──

    def mem_get_info(self) -> Tuple[int, int]:
        """Get free and total memory for the current device. Returns (free, total)."""
        free_mem = c_size_t(0)
        total_mem = c_size_t(0)
        self._check(
            self._lib.hipMemGetInfo(byref(free_mem), byref(total_mem)),
            "hipMemGetInfo",
        )
        return free_mem.value, total_mem.value

    def device_total_mem(self, device: int = 0) -> int:
        """Get total memory for a specific device."""
        total = c_size_t(0)
        self._check(
            self._lib.hipDeviceTotalMem(byref(total), device),
            "hipDeviceTotalMem",
        )
        return total.value

    def get_device_properties_total_mem(self, device: int = 0) -> int:
        """Get totalGlobalMem from hipGetDeviceProperties for a specific device.

        hipDeviceProp_t is a large struct (~800 bytes). We only need totalGlobalMem
        at offset 288 (after name[256] + uuid[16] + luid[8] + luidDeviceNodeMask[4] + pad[4]).
        Allocate a buffer large enough for the full struct and read the field.
        """
        buf = (c_char * 4096)()  # hipDeviceProp_t is ~800 bytes; oversized for safety
        self._check(
            self._lib.hipGetDeviceProperties(byref(buf), device),
            "hipGetDeviceProperties",
        )
        # totalGlobalMem is a size_t at offset 288
        total_global_mem = c_size_t.from_buffer_copy(buf, 288)
        return total_global_mem.value

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

    # --- Mipmapped array allocation wrappers ---

    def malloc_mipmapped_array(self, width: int, height: int = 0, depth: int = 0,
                                num_levels: int = 1,
                                desc_x: int = 32, desc_y: int = 0,
                                desc_z: int = 0, desc_w: int = 0,
                                flags: int = 0) -> int:
        """Allocate a mipmapped HIP array. Returns hipMipmappedArray_t handle."""
        array = c_void_p(0)
        desc = HipChannelFormatDesc(x=desc_x, y=desc_y, z=desc_z, w=desc_w, f=2)
        extent = HIPExtent(width=width, height=height, depth=depth)
        self._check(
            self._lib.hipMallocMipmappedArray(byref(array), byref(desc), extent, num_levels, flags),
            "hipMallocMipmappedArray",
        )
        return array.value or 0

    def mipmapped_array_create(self, width: int, height: int = 0, depth: int = 0,
                                num_levels: int = 1,
                                fmt: int = HIP_AD_FORMAT_FLOAT, num_channels: int = 1,
                                flags: int = 0) -> int:
        """Create a mipmapped HIP array via driver API. Returns hipMipmappedArray_t handle."""
        array = c_void_p(0)
        desc = HipArray3DDescriptor(Width=width, Height=height, Depth=depth,
                                    Format=fmt, NumChannels=num_channels, Flags=flags)
        self._check(
            self._lib.hipMipmappedArrayCreate(byref(array), byref(desc), num_levels),
            "hipMipmappedArrayCreate",
        )
        return array.value or 0

    def free_mipmapped_array(self, array: int) -> None:
        """Free a mipmapped HIP array via hipFreeMipmappedArray."""
        self._check(self._lib.hipFreeMipmappedArray(c_void_p(array)), "hipFreeMipmappedArray")

    def mipmapped_array_destroy(self, array: int) -> None:
        """Destroy a mipmapped HIP array via hipMipmappedArrayDestroy (driver API)."""
        self._check(self._lib.hipMipmappedArrayDestroy(c_void_p(array)), "hipMipmappedArrayDestroy")
