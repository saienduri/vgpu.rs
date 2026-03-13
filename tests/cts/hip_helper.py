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
from ctypes import POINTER, byref, c_int, c_size_t, c_uint, c_void_p
from typing import Optional, Tuple


# ── HIP error codes ──

HIP_SUCCESS = 0
HIP_ERROR_INVALID_VALUE = 1
HIP_ERROR_OUT_OF_MEMORY = 2
HIP_ERROR_NOT_INITIALIZED = 3


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
