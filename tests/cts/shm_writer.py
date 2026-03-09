"""
SHM Writer — Creates binary-compatible SHM files matching Rust's SharedDeviceState V2 layout.

The binary format must match the #[repr(C)] enum SharedDeviceState { V1(...), V2(...) }
defined in vgpu.rs/crates/utils/src/shared_memory/mod.rs and the Go writer in
tensor-fusion/internal/hypervisor/worker/state/soft_limiter_shm.go.

Binary layout (V2):
  [0..4)     u32 LE discriminant = 1 (V2)
  [4..8)     4 bytes alignment padding
  [8..)      SharedDeviceStateV2 data:
               devices:       16 * DeviceEntryV2 (16 * 136 = 2176 bytes)
               device_count:  u32 (4 bytes)
               _pad:          4 bytes alignment
               last_heartbeat: u64 (8 bytes)
               pids:          ShmMutex<Set<usize, 2048>> (32792 bytes, zeroed)
               _padding:      512 bytes

Total file size: 35504 bytes

DeviceEntryV2 (136 bytes):
  uuid:        [u8; 64]  — null-terminated UTF-8 string
  device_info: SharedDeviceInfoV2 (64 bytes)
  is_active:   u32
  _pad:        4 bytes (implicit from repr(C) alignment)

SharedDeviceInfoV2 (64 bytes):
  up_limit:               u32  (offset +0)
  _pad:                   4 bytes (alignment for u64)
  mem_limit:              u64  (offset +8)
  total_cuda_cores:       u32  (offset +16)
  _pad:                   4 bytes (alignment for u64)
  pod_memory_used:        u64  (offset +24)
  erl_token_refill_rate:  u64  (offset +32)  — f64 stored as bits
  erl_token_capacity:     u64  (offset +40)  — f64 stored as bits
  erl_current_tokens:     u64  (offset +48)  — f64 stored as bits
  erl_last_token_update:  u64  (offset +56)  — f64 stored as bits
"""

import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Constants matching Go and Rust definitions ──

MAX_DEVICES = 16
MAX_UUID_LEN = 64
MAX_PROCESSES = 2048

# Rust enum discriminant: V1=0, V2=1
RUST_V2_DISCRIMINANT = 1

# 4-byte discriminant + 4-byte alignment padding
RUST_ENUM_HEADER_SIZE = 8

# DeviceEntryV2 = UUID(64) + SharedDeviceInfoV2(64) + IsActive(4) + pad(4)
RUST_DEVICE_ENTRY_V2_SIZE = 136

# SharedDeviceInfoV2 size (all fields)
RUST_DEVICE_INFO_V2_SIZE = 64

# Offset of pids field within SharedDeviceStateV2
# = devices(16 * 136) + device_count(4) + pad(4) + last_heartbeat(8)
RUST_V2_PIDS_OFFSET = MAX_DEVICES * RUST_DEVICE_ENTRY_V2_SIZE + 4 + 4 + 8

# ShmMutex<Set<usize, 2048>>:
# lock: AtomicUsize(8) + Set<usize,2048>(16384 + 16384 + 8) + pid: usize(8)
RUST_SHM_MUTEX_SET_SIZE = 8 + (MAX_PROCESSES * 8 + MAX_PROCESSES * 8 + 8) + 8

# Padding at end of SharedDeviceStateV2
RUST_STATE_PADDING_SIZE = 512

# Total file size
RUST_SHARED_DEVICE_STATE_TOTAL_SIZE = (
    RUST_ENUM_HEADER_SIZE
    + RUST_V2_PIDS_OFFSET
    + RUST_SHM_MUTEX_SET_SIZE
    + RUST_STATE_PADDING_SIZE
)

assert RUST_SHARED_DEVICE_STATE_TOTAL_SIZE == 35504, (
    f"Total size mismatch: {RUST_SHARED_DEVICE_STATE_TOTAL_SIZE} != 35504"
)

# ── Offsets within the file ──

# Offset of devices array within the file (after enum header)
DEVICES_OFFSET = RUST_ENUM_HEADER_SIZE

# Offset of device_count within the file
DEVICE_COUNT_OFFSET = RUST_ENUM_HEADER_SIZE + MAX_DEVICES * RUST_DEVICE_ENTRY_V2_SIZE

# Offset of last_heartbeat within the file
# device_count(4) + pad(4) = 8 bytes between device_count and last_heartbeat
HEARTBEAT_OFFSET = DEVICE_COUNT_OFFSET + 4 + 4


@dataclass
class DeviceSpec:
    """Specification for a single device in the SHM file."""

    uuid: str
    mem_limit: int  # bytes
    up_limit: int = 80  # percentage (0-100)
    total_cuda_cores: int = 2048
    pod_memory_used: int = 0
    erl_token_refill_rate: float = 10.0
    erl_token_capacity: float = 100.0
    erl_current_tokens: float = 100.0
    erl_last_token_update: float = 0.0
    is_active: bool = True
    device_idx: Optional[int] = None  # If None, auto-assigned sequentially


def _pack_uuid(uuid_str: str) -> bytes:
    """Pack a UUID string into a 64-byte null-terminated buffer."""
    encoded = uuid_str.encode("utf-8")
    if len(encoded) >= MAX_UUID_LEN:
        encoded = encoded[: MAX_UUID_LEN - 1]
    return encoded.ljust(MAX_UUID_LEN, b"\x00")


def _pack_device_info_v2(spec: DeviceSpec) -> bytes:
    """Pack SharedDeviceInfoV2 into 64 bytes.

    Layout (all little-endian):
      u32 up_limit
      u32 _pad (alignment)
      u64 mem_limit
      u32 total_cuda_cores
      u32 _pad (alignment)
      u64 pod_memory_used
      u64 erl_token_refill_rate  (f64 bits)
      u64 erl_token_capacity     (f64 bits)
      u64 erl_current_tokens     (f64 bits)
      u64 erl_last_token_update  (f64 bits)
    """
    return struct.pack(
        "<I I Q I I Q Q Q Q Q",
        spec.up_limit,
        0,  # padding
        spec.mem_limit,
        spec.total_cuda_cores,
        0,  # padding
        spec.pod_memory_used,
        struct.unpack("<Q", struct.pack("<d", spec.erl_token_refill_rate))[0],
        struct.unpack("<Q", struct.pack("<d", spec.erl_token_capacity))[0],
        struct.unpack("<Q", struct.pack("<d", spec.erl_current_tokens))[0],
        struct.unpack("<Q", struct.pack("<d", spec.erl_last_token_update))[0],
    )


def _pack_device_entry_v2(spec: DeviceSpec) -> bytes:
    """Pack a DeviceEntryV2 into 136 bytes.

    Layout:
      [u8; 64]           uuid
      SharedDeviceInfoV2  device_info (64 bytes)
      u32                is_active
      [u8; 4]            padding (repr(C) alignment)
    """
    uuid_bytes = _pack_uuid(spec.uuid)
    info_bytes = _pack_device_info_v2(spec)
    is_active = 1 if spec.is_active else 0
    tail = struct.pack("<I", is_active) + b"\x00" * 4  # is_active + padding

    entry = uuid_bytes + info_bytes + tail
    assert len(entry) == RUST_DEVICE_ENTRY_V2_SIZE, (
        f"DeviceEntryV2 size {len(entry)} != {RUST_DEVICE_ENTRY_V2_SIZE}"
    )
    return entry


def create_shm_file(path: str, devices: List[DeviceSpec]) -> None:
    """Create a SHM file matching the Rust SharedDeviceState V2 binary layout.

    Args:
        path: File path to write (this becomes the 'shm' file the crate mmaps).
        devices: List of device specifications. Max 16 devices.

    The file is created with size RUST_SHARED_DEVICE_STATE_TOTAL_SIZE (35504 bytes).
    All regions not explicitly written (pids, padding) are zero-filled, which is
    the correct initial state (lock=0 means unlocked).
    """
    if len(devices) > MAX_DEVICES:
        raise ValueError(f"Too many devices: {len(devices)} > {MAX_DEVICES}")

    # Start with a zero-filled buffer
    buf = bytearray(RUST_SHARED_DEVICE_STATE_TOTAL_SIZE)

    # Write enum discriminant (V2 = 1)
    struct.pack_into("<I", buf, 0, RUST_V2_DISCRIMINANT)
    # Bytes 4..8 are padding (already zero)

    # Write device entries
    for i, spec in enumerate(devices):
        idx = spec.device_idx if spec.device_idx is not None else i
        if idx >= MAX_DEVICES:
            raise ValueError(f"Device index {idx} >= {MAX_DEVICES}")
        offset = DEVICES_OFFSET + idx * RUST_DEVICE_ENTRY_V2_SIZE
        entry_bytes = _pack_device_entry_v2(spec)
        buf[offset : offset + RUST_DEVICE_ENTRY_V2_SIZE] = entry_bytes

    # Write device_count (u32)
    struct.pack_into("<I", buf, DEVICE_COUNT_OFFSET, len(devices))

    # Write last_heartbeat (u64) — current unix timestamp
    heartbeat = int(time.time())
    struct.pack_into("<Q", buf, HEARTBEAT_OFFSET, heartbeat)

    # pids region and padding are already zeroed — correct initial state

    # Write to file
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf)


def update_heartbeat(path: str, timestamp: Optional[int] = None) -> None:
    """Update the last_heartbeat field in an existing SHM file.

    Args:
        path: Path to the SHM file.
        timestamp: Unix timestamp to write. Defaults to current time.
    """
    if timestamp is None:
        timestamp = int(time.time())

    with open(path, "r+b") as f:
        f.seek(HEARTBEAT_OFFSET)
        f.write(struct.pack("<Q", timestamp))


def read_heartbeat(path: str) -> int:
    """Read the last_heartbeat value from an existing SHM file.

    Returns:
        The heartbeat timestamp as an integer.
    """
    with open(path, "rb") as f:
        f.seek(HEARTBEAT_OFFSET)
        return struct.unpack("<Q", f.read(8))[0]


def read_pod_memory_used(path: str, device_idx: int) -> int:
    """Read pod_memory_used for a specific device from the SHM file.

    Args:
        path: Path to the SHM file.
        device_idx: Index of the device (0-15).

    Returns:
        The pod_memory_used value in bytes.
    """
    if device_idx >= MAX_DEVICES:
        raise ValueError(f"Device index {device_idx} >= {MAX_DEVICES}")

    # pod_memory_used is at offset 24 within SharedDeviceInfoV2
    # SharedDeviceInfoV2 starts at offset 64 within DeviceEntryV2 (after uuid)
    device_offset = DEVICES_OFFSET + device_idx * RUST_DEVICE_ENTRY_V2_SIZE
    info_offset = device_offset + MAX_UUID_LEN  # skip uuid
    pod_mem_offset = info_offset + 24  # up_limit(4) + pad(4) + mem_limit(8) + total_cores(4) + pad(4) = 24

    with open(path, "rb") as f:
        f.seek(pod_mem_offset)
        return struct.unpack("<Q", f.read(8))[0]


def read_mem_limit(path: str, device_idx: int) -> int:
    """Read mem_limit for a specific device from the SHM file.

    Args:
        path: Path to the SHM file.
        device_idx: Index of the device (0-15).

    Returns:
        The mem_limit value in bytes.
    """
    if device_idx >= MAX_DEVICES:
        raise ValueError(f"Device index {device_idx} >= {MAX_DEVICES}")

    device_offset = DEVICES_OFFSET + device_idx * RUST_DEVICE_ENTRY_V2_SIZE
    info_offset = device_offset + MAX_UUID_LEN
    mem_limit_offset = info_offset + 8  # up_limit(4) + pad(4) = 8

    with open(path, "rb") as f:
        f.seek(mem_limit_offset)
        return struct.unpack("<Q", f.read(8))[0]


def read_device_count(path: str) -> int:
    """Read the device_count from the SHM file."""
    with open(path, "rb") as f:
        f.seek(DEVICE_COUNT_OFFSET)
        return struct.unpack("<I", f.read(4))[0]


def read_device_uuid(path: str, device_idx: int) -> str:
    """Read the UUID for a specific device from the SHM file."""
    if device_idx >= MAX_DEVICES:
        raise ValueError(f"Device index {device_idx} >= {MAX_DEVICES}")

    device_offset = DEVICES_OFFSET + device_idx * RUST_DEVICE_ENTRY_V2_SIZE

    with open(path, "rb") as f:
        f.seek(device_offset)
        uuid_bytes = f.read(MAX_UUID_LEN)

    # Find null terminator
    null_pos = uuid_bytes.find(b"\x00")
    if null_pos >= 0:
        uuid_bytes = uuid_bytes[:null_pos]
    return uuid_bytes.decode("utf-8")
