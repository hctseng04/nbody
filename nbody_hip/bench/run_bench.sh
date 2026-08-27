#!/usr/bin/env bash
# Shared runner behind bench_{1,2,4,5}gpu.sh -- runs
#   nbody_hip -benchmark -numdevices=<N> -numbodies=<NUMBODIES> -fp64
# and tees the full output to its own timestamped log file under logs/.
#
# Usage: run_bench.sh <numdevices> [extra args passed through to nbody_hip]
#
# Env overrides:
#   NBODY_BIN   path to the nbody_hip binary (default: ../build/nbody_hip)
#   NUMBODIES   body count (default: 2097152, matching the requested benchmark)
#   LOG_DIR     where to write logs (default: ./logs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${NBODY_BIN:-$SCRIPT_DIR/../build/nbody_hip}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
NUMBODIES="${NUMBODIES:-2097152}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <numdevices> [extra nbody_hip args...]" >&2
    exit 1
fi

NUMDEVICES="$1"
shift

if [ ! -x "$BIN" ]; then
    echo "Error: nbody_hip binary not found/executable at: $BIN" >&2
    echo "Build it first (see README.md), or set NBODY_BIN=/path/to/nbody_hip" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/nbody_fp64_${NUMDEVICES}gpu_${TIMESTAMP}.log"

echo "=== nbody_hip: ${NUMDEVICES} GPU(s), numbodies=${NUMBODIES}, fp64 ==="
echo "Binary : $BIN"
echo "Log    : $LOG_FILE"

{
    echo "# command: $BIN -benchmark -numdevices=${NUMDEVICES} -numbodies=${NUMBODIES} -fp64 $*"
    echo "# host: $(hostname)"
    echo "# date: $(date -Iseconds)"
    echo "# GPU inventory:"
    rocm-smi --showproductname 2>/dev/null || echo "  (rocm-smi unavailable)"
    echo "----"
    /usr/bin/time -v "$BIN" -benchmark -numdevices="${NUMDEVICES}" -numbodies="${NUMBODIES}" -fp64 "$@" 2>&1 \
        || echo "# nbody_hip exited with status $?"
} | tee "$LOG_FILE"

echo "Wrote $LOG_FILE"
