#!/bin/bash
# Test hip-limiter memory enforcement in a client pod.
# Usage: bash test-mem-enforcement.sh
#   or:  kubectl exec -it <pod> -- bash test-mem-enforcement.sh

set -euo pipefail

python3 -u - <<'PYEOF'
import sys

def gb(b):
    return f"{b / (1024**3):.2f} GiB"

def tensor_size_gib(shape, dtype_bytes=2):
    """Calculate actual tensor size in GiB."""
    n = 1
    for d in shape:
        n *= d
    return n * dtype_bytes / (1024**3)

def mem():
    import torch
    free, total = torch.cuda.mem_get_info(0)
    used = total - free
    return free, total, used

def report(label):
    free, total, used = mem()
    print(f"[{label}] total={gb(total)}  used={gb(used)}  free={gb(free)}")

print("=" * 60)
print("Hip-limiter memory enforcement test")
print("=" * 60)

import torch
print(f"PyTorch {torch.__version__}, device: {torch.cuda.get_device_name(0)}")
print()

# 1. Baseline
report("baseline")
_, total, _ = mem()
limit_gib = total / (1024**3)
print(f"  Reported limit: {limit_gib:.1f} GiB")
print()

# 2. Allocate ~4 GiB  (2048*1024*1024*2 bytes = 4 GiB)
shape_x = (2048, 1024, 1024)
sz = tensor_size_gib(shape_x)
print(f"--- Allocating {sz:.1f} GiB tensor (x) ---")
x = torch.zeros(*shape_x, dtype=torch.float16, device='cuda')
report("after alloc x")
print()

# 3. Allocate ~8 GiB  (4096*1024*1024*2 bytes = 8 GiB)
shape_y = (4096, 1024, 1024)
sz = tensor_size_gib(shape_y)
print(f"--- Allocating {sz:.1f} GiB tensor (y) ---")
y = torch.zeros(*shape_y, dtype=torch.float16, device='cuda')
report("after alloc y")
print()

# 4. Free x, check usage decreases
print("--- Freeing x (del + empty_cache) ---")
del x
torch.cuda.empty_cache()
report("after free x")
print()

# 5. Try to exceed limit: y=8 GiB is still held, try another 10 GiB
shape_z = (5120, 1024, 1024)
sz = tensor_size_gib(shape_z)
print(f"--- Attempting {sz:.1f} GiB allocation (should exceed {limit_gib:.0f} GiB limit) ---")
try:
    z = torch.zeros(*shape_z, dtype=torch.float16, device='cuda')
    print(f"FAIL: allocation should have been denied (8+{sz:.0f}={8+sz:.0f} > {limit_gib:.0f})")
    del z
    torch.cuda.empty_cache()
except RuntimeError as e:
    print(f"PASS: OOM raised: {e}")
report("after denied alloc")
print()

# 6. Free y, check back to baseline
print("--- Freeing y (del + empty_cache) ---")
del y
torch.cuda.empty_cache()
report("after free y")
print()

# 7. Summary
free, total, used = mem()
if used < 1024 * 1024 * 100:  # < 100 MiB
    print("PASS: usage returned to ~0 after freeing all tensors")
else:
    print(f"WARN: {gb(used)} still used after freeing all tensors (caching allocator?)")

print()
print("=" * 60)
print("Done")
print("=" * 60)
PYEOF
