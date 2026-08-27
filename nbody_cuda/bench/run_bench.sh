#!/usr/bin/env bash
# Shared runner behind bench_{1,2,4,5,8}gpu.sh -- runs
#   nbody -benchmark -numdevices=<N> -numbodies=<NUMBODIES> -fp64
# and tees the full output to its own timestamped log file under logs/.
#
# NOTE: this targets the *CUDA* nbody sample (this directory), built with
# nvcc against an NVIDIA CUDA toolkit. It will NOT run on a ROCm/AMD node --
# it needs an actual CUDA-capable machine with the `nbody` binary already
# built (e.g. via `cmake -B build && cmake --build build` from this
# directory, same as the sibling nbody_hip/ port's build flow).
#
# For a HIP/MI210 version of this same sweep, see
# ../../nbody_hip/bench/{run_bench,bench_*gpu,run_all,summarize}.sh --
# this script is its CUDA-side counterpart, same design.
#
# Usage: run_bench.sh <numdevices> [extra args passed through to nbody]
#
# Env overrides:
#   NBODY_BIN   path to the nbody binary (default: ../build/nbody)
#   NUMBODIES   body count (default: 2097152)
#   LOG_DIR     where to write logs (default: ./logs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${NBODY_BIN:-$SCRIPT_DIR/../build/nbody}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
NUMBODIES="${NUMBODIES:-2097152}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <numdevices> [extra nbody args...]" >&2
    exit 1
fi

NUMDEVICES="$1"
shift

if [ ! -x "$BIN" ]; then
    echo "Error: nbody binary not found/executable at: $BIN" >&2
    echo "Build it first (see ../README.md), or set NBODY_BIN=/path/to/nbody" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/nbody_fp64_${NUMDEVICES}gpu_${TIMESTAMP}.log"

echo "=== nbody (CUDA): ${NUMDEVICES} GPU(s), numbodies=${NUMBODIES}, fp64 ==="
echo "Binary : $BIN"
echo "Log    : $LOG_FILE"

TIME_CMD=()
if command -v /usr/bin/time >/dev/null 2>&1; then
    TIME_CMD=(/usr/bin/time -v)
fi

{
    echo "# command: $BIN -benchmark -numdevices=${NUMDEVICES} -numbodies=${NUMBODIES} -fp64 $*"
    echo "# host: $(hostname)"
    echo "# date: $(date -Iseconds)"
    echo "# GPU inventory:"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L
        echo
        nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total --format=csv
    else
        echo "  (nvidia-smi not found)"
    fi
    echo "----"
    "${TIME_CMD[@]}" "$BIN" -benchmark -numdevices="${NUMDEVICES}" -numbodies="${NUMBODIES}" -fp64 "$@" 2>&1 \
        || echo "# nbody exited with status $?"
} | tee "$LOG_FILE"

echo "Wrote $LOG_FILE"
