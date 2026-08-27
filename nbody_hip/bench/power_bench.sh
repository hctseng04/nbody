#!/usr/bin/env bash
# Runs nbody_hip -benchmark for <numdevices> GPUs while sampling power draw
# via amd-smi every SAMPLE_MS milliseconds across the run, then averages --
# deliberately the SAME sampling methodology (same interval, same averaging
# formula) as the CUDA-side nbody_cuda/bench/power_bench.sh, so the two
# platforms' numbers are directly, apples-to-apples comparable.
#
# NOTE: amd-smi also exposes a more precise hardware energy-consumption
# *counter* (`amd-smi metric -E`), which an earlier version of this script
# used (read once before/after the run -- an exact delta, no sampling
# error). NVIDIA's driver/GPU on the other host has no equivalent counter
# exposed at all, so that approach can't be matched on both sides. This
# version trades away that extra AMD-side precision for a fair, identical
# method across platforms; if you only care about the MI210 number and want
# the more precise reading, use the energy-counter delta instead (see git
# history for the previous version of this script).
#
# Usage: power_bench.sh <numdevices> [extra nbody_hip args]
#
# Env overrides: NBODY_BIN, NUMBODIES, LOG_DIR (same as run_bench.sh), SAMPLE_MS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${NBODY_BIN:-$SCRIPT_DIR/../build/nbody_hip}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
NUMBODIES="${NUMBODIES:-2097152}"
SAMPLE_MS="${SAMPLE_MS:-200}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <numdevices> [extra nbody_hip args...]" >&2
    exit 1
fi
NUMDEVICES="$1"
shift

if [ ! -x "$BIN" ]; then
    echo "Error: nbody_hip binary not found/executable at: $BIN" >&2
    exit 1
fi
if ! command -v amd-smi >/dev/null 2>&1; then
    echo "Error: amd-smi not found -- required for power measurement" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/nbody_fp64_power_${NUMDEVICES}gpu_${TIMESTAMP}.log"
SAMPLE_FILE=$(mktemp)
SLEEP_SEC=$(awk -v ms="$SAMPLE_MS" 'BEGIN{printf "%.3f", ms/1000}')

{
    echo "=== nbody_hip (power): ${NUMDEVICES} GPU(s), numbodies=${NUMBODIES}, fp64 ==="
    echo "# command: $BIN -benchmark -numdevices=${NUMDEVICES} -numbodies=${NUMBODIES} -fp64 $*"
    echo "# host: $(hostname)"
    echo "# date: $(date -Iseconds)"
    echo "# method: amd-smi socket_power sampled every ${SAMPLE_MS}ms on GPUs 0-$((NUMDEVICES - 1)) for the run's duration, averaged per GPU then summed -- same method as the CUDA-side script"
    echo "----"

    (
        while true; do
            amd-smi metric -p --csv 2>/dev/null | tail -n +2 \
                | awk -F, -v n="$NUMDEVICES" 'BEGIN{OFS=","} { gsub(/\r/,"",$0); if ($1+0 < n) print $1,$2 }'
            sleep "$SLEEP_SEC"
        done
    ) > "$SAMPLE_FILE" &
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
