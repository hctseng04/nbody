#!/usr/bin/env bash
# Runs nbody -benchmark for <numdevices> GPUs while sampling per-GPU power
# draw via nvidia-smi every SAMPLE_MS milliseconds across the run, then
# averages -- deliberately the SAME sampling methodology (same interval,
# same averaging formula) as the HIP-side nbody_hip/bench/power_bench.sh,
# so the two platforms' numbers are directly, apples-to-apples comparable.
#
# NOTE: this driver/GPU combination doesn't expose a cumulative hardware
# energy counter (AMD's amd-smi does, via `-E`), so sampling is the only
# method available here -- the HIP-side script was brought down to sampling
# too, rather than leaving it on the more precise counter, so both sides
# use an identical method instead of an apples-to-oranges comparison.
#
# Usage: power_bench.sh <numdevices> [extra nbody args]
#
# Env overrides: NBODY_BIN, NUMBODIES, LOG_DIR (same as run_bench.sh), SAMPLE_MS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${NBODY_BIN:-$SCRIPT_DIR/../build/nbody}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
NUMBODIES="${NUMBODIES:-2097152}"
SAMPLE_MS="${SAMPLE_MS:-200}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <numdevices> [extra nbody args...]" >&2
    exit 1
fi
NUMDEVICES="$1"
shift

if [ ! -x "$BIN" ]; then
    echo "Error: nbody binary not found/executable at: $BIN" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi not found -- required for power measurement" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/nbody_fp64_power_${NUMDEVICES}gpu_${TIMESTAMP}.log"
SAMPLE_FILE=$(mktemp)
GPU_LIST=$(seq -s, 0 $((NUMDEVICES - 1)))

{
    echo "=== nbody (power): ${NUMDEVICES} GPU(s), numbodies=${NUMBODIES}, fp64 ==="
    echo "# command: $BIN -benchmark -numdevices=${NUMDEVICES} -numbodies=${NUMBODIES} -fp64 $*"
    echo "# host: $(hostname)"
    echo "# date: $(date -Iseconds)"
    echo "# method: nvidia-smi power.draw sampled every ${SAMPLE_MS}ms on GPUs ${GPU_LIST} for the run's duration, averaged per GPU then summed -- same method as the HIP-side script"
    echo "----"

    nvidia-smi --query-gpu=index,power.draw --format=csv,noheader,nounits \
        -i "$GPU_LIST" -lms "$SAMPLE_MS" > "$SAMPLE_FILE" 2>/dev/null &
    SAMPLER_PID=$!
    sleep 0.3   # let the sampler take at least one reading before the run starts

    T0=$(date +%s.%N)
    "$BIN" -benchmark -numdevices="${NUMDEVICES}" -numbodies="${NUMBODIES}" -fp64 "$@" 2>&1 | tee /tmp/.power_bench_out.$$
    T1=$(date +%s.%N)

    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true

    ELAPSED=$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.3f", b-a}')
    N_SAMPLES=$(wc -l < "$SAMPLE_FILE")

    # sum($2)/count = average power PER GPU per sample row; multiply by
    # NUMDEVICES to get average TOTAL system power across all GPUs used.
    AVG_PER_GPU=$(awk -F, 'NF>=2 { gsub(/[ \r]/,"",$2); if ($2 != "") { sum+=$2; n++ } } END{ if(n>0) printf "%.2f", sum/n; else print "0" }' "$SAMPLE_FILE")
    AVG_POWER=$(awk -v p="$AVG_PER_GPU" -v n="$NUMDEVICES" 'BEGIN{printf "%.2f", p*n}')
    TOTAL_J=$(awk -v p="$AVG_POWER" -v t="$ELAPSED" 'BEGIN{printf "%.3f", p*t}')

    GFLOPS=$(grep -oP '= \K[0-9.]+(?= double-precision GFLOP/s)' /tmp/.power_bench_out.$$ | tail -1 || true)
    rm -f /tmp/.power_bench_out.$$ "$SAMPLE_FILE"

    if [ -n "${GFLOPS:-}" ]; then
        PER_WATT=$(awk -v g="$GFLOPS" -v p="$AVG_POWER" 'BEGIN{ if(p>0) printf "%.2f", g/p; else print "n/a" }')
    else
        PER_WATT="n/a"
    fi

    echo "----"
    echo "wall-clock measurement window: ${ELAPSED} s"
    echo "power samples collected: ${N_SAMPLES} rows (across ${NUMDEVICES} GPU(s))"
    echo "average total power draw: ${AVG_POWER} W"
    echo "estimated total GPU energy consumed: ${TOTAL_J} J"
    echo "performance per watt: ${PER_WATT} GFLOP/s per W"
} | tee "$LOG_FILE"

echo "Wrote $LOG_FILE"
