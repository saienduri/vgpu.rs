#!/usr/bin/env bash
# run-cts.sh — Build and run the hip-limiter Conformance Test Suite
#
# Usage:
#   bash scripts/run-cts.sh              # Run full suite (Rust fuzzer + fast CTS)
#   bash scripts/run-cts.sh --fuzz-only  # Run only the Rust fuzzer (no GPU needed)
#   bash scripts/run-cts.sh --cts-only   # Run only the CTS (needs MI325X + Docker)
#   bash scripts/run-cts.sh --full       # Use full Dockerfile (slow, clean build in Docker)
#   bash scripts/run-cts.sh --cts-only -- -k test_alloc_within_limit  # Pass args to pytest
#
# Fast mode (default): builds hip-limiter natively (~3s incremental), runs all
# 105 tests in a thin Docker container with ROCm + .so + tests volume-mounted (~30s).
#
# Full mode (--full): uses the original Dockerfile with everything baked in.
# Same 105 tests but slower (~3 min) due to clean cargo build in Docker.
#
# One-time setup for fast mode:
#   # On remote: install Rust, extract TheRock, build thin image
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#   sudo docker cp $(docker create hip-limiter-cts):/opt/rocm-7.11.0rc0 /opt/rocm-7.11.0rc0
#   sudo ln -sf /opt/rocm-7.11.0rc0 /opt/therock
#   docker build -t cts-runner -f tests/cts/Dockerfile.runner .

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

run_fuzz=true
run_cts=true
full_mode=false
pytest_args=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fuzz-only) run_cts=false; shift ;;
        --cts-only)  run_fuzz=false; shift ;;
        --full)      full_mode=true; shift ;;
        --)          shift; pytest_args=("$@"); break ;;
        *)           pytest_args+=("$1"); shift ;;
    esac
done

echo "========================================="
echo " hip-limiter Test Suite"
echo "========================================="

fuzz_passed=true
cts_passed=true

# ── Phase 1: Rust Fuzzer (no GPU needed) ──

if $run_fuzz; then
    echo ""
    echo -e "${YELLOW}[1/2] Running Rust fuzzer (proptest)...${NC}"
    echo ""
    cd "$PROJECT_DIR"
    if cargo test -p hip-limiter-fuzz 2>&1; then
        echo ""
        echo -e "${GREEN}[1/2] Rust fuzzer: PASSED${NC}"
    else
        echo ""
        echo -e "${RED}[1/2] Rust fuzzer: FAILED${NC}"
        fuzz_passed=false
    fi
fi

# ── Phase 2: Python CTS in Docker (needs MI325X) ──

if $run_cts; then
    echo ""
    cd "$PROJECT_DIR"

    # Default pytest args if none provided
    if [[ ${#pytest_args[@]} -eq 0 ]]; then
        pytest_args=("-v" "tests/cts/")
    fi

    if $full_mode; then
        # Full mode: build everything in Docker (slow, all tests including SMI)
        echo -e "${YELLOW}[2/2] Building full CTS Docker image...${NC}"
        if ! docker build -t hip-limiter-cts -f tests/cts/Dockerfile . ; then
            echo -e "${RED}[2/2] Docker build: FAILED${NC}"
            cts_passed=false
        else
            echo ""
            echo -e "${YELLOW}[2/2] Running Python CTS on GPU (full mode)...${NC}"
            echo ""
            if docker run --rm \
                --device=/dev/kfd --device=/dev/dri --group-add video \
                hip-limiter-cts pytest "${pytest_args[@]}"; then
                echo -e "${GREEN}[2/2] Python CTS: PASSED${NC}"
            else
                echo -e "${RED}[2/2] Python CTS: FAILED${NC}"
                cts_passed=false
            fi
        fi
    else
        # Fast mode: build natively, run in thin container with volume mounts
        echo -e "${YELLOW}[2/2] Building hip-limiter (native, incremental)...${NC}"
        if ! cargo build --release -p hip-limiter; then
            echo -e "${RED}[2/2] Cargo build: FAILED${NC}"
            cts_passed=false
        else
            SO_PATH="${PROJECT_DIR}/target/release/libhip_limiter.so"
            if [[ ! -f "$SO_PATH" ]]; then
                echo -e "${RED}[2/2] Build succeeded but $SO_PATH not found${NC}"
                cts_passed=false
            else

            THEROCK_PATH="${THEROCK_PATH:-/opt/therock}"
            if [[ ! -d "$THEROCK_PATH" ]]; then
                echo -e "${RED}[2/2] ROCm not found at $THEROCK_PATH${NC}"
                echo -e "${RED}       Set THEROCK_PATH to your ROCm installation${NC}"
                cts_passed=false
            else

            # Build thin runner if it doesn't exist
            if ! docker image inspect cts-runner &>/dev/null; then
                echo ""
                echo -e "${YELLOW}[2/2] Building CTS runner image (first time)...${NC}"
                docker build -t cts-runner -f tests/cts/Dockerfile.runner .
            fi

            echo ""
            echo -e "${YELLOW}[2/2] Running Python CTS on GPU (fast mode)...${NC}"
            echo ""

            if docker run --rm \
                --device=/dev/kfd --device=/dev/dri --group-add video \
                -v "${THEROCK_PATH}:/opt/rocm:ro" \
                -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:ro \
                -v "${SO_PATH}:/workspace/libhip_limiter.so:ro" \
                -v "${PROJECT_DIR}/tests/cts:/workspace/tests/cts:ro" \
                cts-runner pytest "${pytest_args[@]}"; then
                echo ""
                echo -e "${GREEN}[2/2] Python CTS: PASSED${NC}"
            else
                echo ""
                echo -e "${RED}[2/2] Python CTS: FAILED${NC}"
                cts_passed=false
            fi

            fi # SO_PATH check
            fi # THEROCK_PATH check
        fi
    fi
fi

# ── Summary ──

echo ""
echo "========================================="
echo " Summary"
echo "========================================="

if $run_fuzz; then
    if $fuzz_passed; then
        echo -e "  Rust fuzzer:  ${GREEN}PASSED${NC}"
    else
        echo -e "  Rust fuzzer:  ${RED}FAILED${NC}"
    fi
fi

if $run_cts; then
    if $cts_passed; then
        echo -e "  Python CTS:   ${GREEN}PASSED${NC}"
    else
        echo -e "  Python CTS:   ${RED}FAILED${NC}"
    fi
fi

echo "========================================="

if $fuzz_passed && $cts_passed; then
    exit 0
else
    exit 1
fi
