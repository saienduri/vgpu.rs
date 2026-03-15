# Remaining HIP Allocation Hook Coverage

> Status: Tasks 1, 2, 3, 4, 7 completed. Tasks 3 pushed (`f3dd55c`), Task 7 pending CTS validation.
> Tests: 128 Rust all green. CTS pending remote run for Task 7.

## Current Hook Coverage (24 hooks)

**Alloc (13):** hipMalloc, hipHostMalloc, hipHostAlloc, hipMallocHost, hipMemAllocHost, hipMallocManaged, hipExtMallocWithFlags, hipMallocAsync, hipMallocFromPoolAsync, hipMallocPitch, hipMemAllocPitch, hipMalloc3D, hipMemCreate

**Free (5):** hipFree, hipHostFree, hipFreeHost, hipFreeAsync, hipMemRelease

**Spoofing (5):** hipMemGetInfo, hipDeviceTotalMem, rsmi_dev_memory_total_get, amdsmi_get_gpu_memory_total, amdsmi_get_gpu_vram_info

## Remaining Tasks

### Task 3: hipMemAllocPitch (Medium priority)

Driver API version of hipMallocPitch. Same two-phase pitched pattern.

- **Signature:** `hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height, unsigned int elementSizeBytes)`
- **Approach:** Reuse `check_and_alloc_pitched!` macro — identical reserve/native/overhead flow
- **Key difference:** Has `elementSizeBytes` param that influences pitch alignment, but doesn't change accounting logic
- **Files:** `crates/hip-limiter/src/detour/mem.rs` (hook + attach), `tests/cts/test_allocation_enforcement.py` (variant test), `tests/cts/hip_helper.py` (wrapper)
- **Look up signature in:** `refs/rocm-systems/projects/hip/include/hip/hip_runtime_api.h`

### Task 5: Array Family (Low priority)

Array allocations — rare in ML workloads, mostly graphics/texture operations.

- `hipMallocArray(hipArray_t* array, const hipChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags)`
- `hipMalloc3DArray(hipArray_t* array, const hipChannelFormatDesc* desc, hipExtent extent, unsigned int flags)`
- `hipArrayCreate(hipArray_t* array, const HIP_ARRAY_DESCRIPTOR* desc)`
- `hipArray3DCreate(hipArray_t* array, const HIP_ARRAY3D_DESCRIPTOR* desc)`
- `hipFreeArray(hipArray_t array)`
- `hipArrayDestroy(hipArray_t array)` (alias of hipFreeArray)

**Challenge:** Size computation requires parsing `hipChannelFormatDesc` (bits per channel x num channels) and multiplying by dimensions. Need new ctypes structs in hip_helper.py.

### Task 6: Mipmapped Arrays (Low priority)

Very rare, graphics-only.

- `hipMipmappedArrayCreate(hipMipmappedArray_t* mipmappedArray, const HIP_ARRAY3D_DESCRIPTOR* desc, unsigned int numMipmapLevels)`
- `hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray)`
- `hipMipmappedArrayGetLevel(hipArray_t* levelArray, hipMipmappedArray_t mipmappedArray, unsigned int level)`

### Task 7: Virtual Memory Management (Low priority)

Advanced use case — explicit virtual address reservation + physical backing.

- `hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size, const hipMemAllocationProp* prop, unsigned long long flags)`
- `hipMemRelease(hipMemGenericAllocationHandle_t handle)`

**Challenge:** Two-step model (create handle, then map). Tracking is different — need to track handle->size, not pointer->size.

## Decision Context

Tasks 5-7 are unlikely to be hit by ML frameworks (PyTorch, JAX, TensorFlow). Task 3 is a straightforward addition using existing infrastructure. Consider whether the coverage from 20 hooks is sufficient for the deployment target before investing in niche APIs.
