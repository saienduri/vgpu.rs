# HIP Memory Hook Coverage

All HIP memory allocation APIs that consume physical VRAM are hooked. This document tracks what is hooked, what was evaluated and excluded, and known gaps.

## Hook Coverage (34 hooks)

**Alloc (19):** hipMalloc, hipExtMallocWithFlags, hipHostMalloc, hipHostAlloc, hipMallocHost, hipMemAllocHost, hipMallocManaged, hipMallocAsync, hipMallocFromPoolAsync, hipMallocPitch, hipMemAllocPitch, hipMalloc3D, hipMemCreate, hipMallocArray, hipMalloc3DArray, hipArrayCreate, hipArray3DCreate, hipMallocMipmappedArray, hipMipmappedArrayCreate

**Free (9):** hipFree, hipHostFree, hipFreeHost, hipFreeAsync, hipMemRelease, hipFreeArray, hipArrayDestroy, hipFreeMipmappedArray, hipMipmappedArrayDestroy

**Spoofing (5):** hipMemGetInfo, hipDeviceTotalMem (inline Frida), rsmi_dev_memory_total_get, amdsmi_get_gpu_memory_total, amdsmi_get_gpu_vram_info (dlsym-level)

**System (1):** dlsym (catches late-loaded libraries)

## Evaluated and correctly NOT hooked

| API | Reason |
|-----|--------|
| `hipMemAddressFree` / `hipMemAddressReserve` / `hipMemMap` / `hipMemUnmap` | VMM address management only — no physical VRAM. Physical backing comes via `hipMemCreate`/`hipMemRelease` which ARE hooked. |
| `hipExternalMemoryGetMappedBuffer` | Maps memory allocated by another API (Vulkan, OpenGL, DMA-buf). No new physical allocation — the foreign API owns the VRAM. |
| `hipMemPoolImportPointer` | Imports a pointer from another process's pool. No new allocation — memory was already allocated and accounted for in the exporting process. |

## Known gap: graph memory nodes

`hipGraphAddMemAllocNode` and `hipGraphAddMemFreeNode` are NOT hooked. These APIs use an internal CLR pool allocator (`MemoryPool::AllocateMemory` → `amd::SvmBuffer::malloc()`) that bypasses all public HIP APIs — neither `hipMalloc` nor `hipFree` is called during graph execution (`hipGraphLaunch`). Physical VRAM is allocated at launch time, retained in a per-device pool across graph lifetimes, and only released to the OS via `hipDeviceGraphMemTrim()`.

**Why this is accepted:**
1. No ML framework (PyTorch, JAX, TensorFlow, ONNX Runtime) uses graph memory alloc nodes. All pre-allocate memory via private pools or arena allocators before graph capture, then capture only kernel launches. The one edge case is cuBLAS/hipBLAS 12+, which can unintentionally produce alloc nodes via internal `cudaMallocAsync` workspace calls during stream capture — frameworks work around this by pre-setting workspace via `cublasSetWorkspace()`.
2. No production GPU limiter (HAMi-core, tkestack/vcuda, NVIDIA MPS) hooks graph memory.
3. Synchronous enforcement is impossible — the internal allocator has no public interception point, and before/after queries on `hipDeviceGetGraphMemAttribute` are racy under concurrent graph launches.

**Monitoring path if this becomes relevant:**
- `hipDeviceGetGraphMemAttribute(hipGraphMemAttrReservedMemCurrent)` queries actual graph pool VRAM usage
- `hipDeviceGraphMemTrim(device)` reclaims inactive pool memory back to OS
- These could be integrated as a periodic reconciliation mechanism without per-node hooking
