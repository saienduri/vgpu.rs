#!/usr/bin/env bash
# run-cts.sh — Build and run the hip-limiter Conformance Test Suite
#
# Usage:
#   bash scripts/run-cts.sh              # Run full suite (Rust fuzzer + Python CTS in Docker)
#   bash scripts/run-cts.sh --fuzz-only  # Run only the Rust fuzzer (no GPU needed)
#   bash scripts/run-cts.sh --cts-only   # Run only the Python CTS (needs MI325X + Docker)
#   bash scripts/run-cts.sh --cts-only -- -k test_alloc_within_limit  # Pass args to pytest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CTS_IMAGE="hip-limiter-cts"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

run_fuzz=true
run_cts=true
pytest_args=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fuzz-only) run_cts=false; shift ;;
        --cts-only)  run_fuzz=false; shift ;;
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
    echo -e "${YELLOW}[2/2] Building CTS Docker image...${NC}"
    echo ""
    cd "$PROJECT_DIR"

    if ! docker build -t "$CTS_IMAGE" -f tests/cts/Dockerfile . ; then
        echo -e "${RED}[2/2] Docker build: FAILED${NC}"
        cts_passed=false
    else
        echo ""
        echo -e "${YELLOW}[2/2] Running Python CTS on GPU...${NC}"
        echo ""

        # Default pytest args if none provided
        if [[ ${#pytest_args[@]} -eq 0 ]]; then
            pytest_args=("-v" "tests/cts/")
        fi

        if docker run --rm \
            --device=/dev/kfd \
            --device=/dev/dri \
            --group-add video \
            "$CTS_IMAGE" \
            pytest "${pytest_args[@]}"; then
            echo ""
            echo -e "${GREEN}[2/2] Python CTS: PASSED${NC}"
        else
            echo ""
            echo -e "${RED}[2/2] Python CTS: FAILED${NC}"
            cts_passed=false
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
