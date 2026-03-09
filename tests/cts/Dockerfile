# CTS Runner — Conformance Test Suite for hip-limiter
#
# Builds hip-limiter from source, installs TheRock 7.11 ROCm, and runs
# pytest-based CTS tests on MI325X hardware.
#
# Build:
#   docker build -t cts-runner -f tests/cts/Dockerfile .
#
# Run (requires AMD GPU passthrough):
#   docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video cts-runner
#
# Run specific test:
#   docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
#     cts-runner pytest -v tests/cts/test_allocation_enforcement.py

FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    build-essential \
    pkg-config \
    libssl-dev \
    llvm-dev \
    libclang-dev \
    clang \
    && rm -rf /var/lib/apt/lists/*

# ── Install TheRock 7.11 (ROCm) ──
# Pre-download the tarball and place it at the build context root:
#   wget -O therock-rocm.tar.gz https://rocm.prereleases.amd.com/tarball/therock-dist-linux-gfx94X-dcgpu-7.11.0rc0.tar.gz

ARG ROCM_VERSION=7.11.0rc0

COPY therock-rocm.tar.gz /tmp/therock-rocm.tar.gz
RUN set -e && \
    mkdir -p "/opt/rocm-${ROCM_VERSION}" && \
    EXTRACT_DIR=$(mktemp -d) && \
    tar -xzf /tmp/therock-rocm.tar.gz -C "${EXTRACT_DIR}" && \
    (mv "${EXTRACT_DIR}"/* "/opt/rocm-${ROCM_VERSION}/" 2>/dev/null || \
     cp -r "${EXTRACT_DIR}"/* "/opt/rocm-${ROCM_VERSION}/") && \
    rm -rf "${EXTRACT_DIR}" /tmp/therock-rocm.tar.gz && \
    ln -sf "/opt/rocm-${ROCM_VERSION}" /opt/rocm && \
    echo "TheRock ${ROCM_VERSION} installed."

ENV ROCM_PATH=/opt/rocm
ENV PATH="${ROCM_PATH}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ROCM_PATH}/lib"

# ── Install Rust ──

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# ── Build hip-limiter ──

COPY . /workspace/vgpu.rs
WORKDIR /workspace/vgpu.rs

RUN cargo build --release -p hip-limiter

ENV HIP_LIMITER_LIB=/workspace/vgpu.rs/target/release/libhip_limiter.so

# ── Install Python test dependencies ──

RUN pip3 install --no-cache-dir -r tests/cts/requirements.txt

# ── Runtime setup ──

# The CTS auto-discovers the test UUID from the first visible GPU.
# Override with CTS_TEST_UUID env var if needed.

WORKDIR /workspace/vgpu.rs
CMD ["pytest", "-v", "tests/cts/"]
