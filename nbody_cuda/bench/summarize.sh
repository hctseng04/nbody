#!/usr/bin/env bash
# Prints a scaling table from the most recent log in logs/ for each of
# 1/2/4/5/8 GPUs: total time, GFLOP/s, speedup and parallel efficiency
# relative to the 1-GPU run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"

printf "%-6s %-14s %-14s %-10s %-12s %s\n" "GPUs" "time_ms" "GFLOP/s" "speedup" "efficiency" "log"

base_gflops=""
for n in 1 2 4 5 8; do
    latest=$(ls -t "$LOG_DIR"/nbody_fp64_${n}gpu_*.log 2>/dev/null | head -1 || true)

    if [ -z "$latest" ]; then
        printf "%-6s %-14s\n" "$n" "(no log found -- run bench_${n}gpu.sh first)"
        continue
    fi

    time_ms=$(grep -oP 'total time for \d+ iterations: \K[0-9.]+' "$latest" | tail -1 || true)
    gflops=$(grep -oP '= \K[0-9.]+(?= double-precision GFLOP/s)' "$latest" | tail -1 || true)

    if [ -z "$time_ms" ] || [ -z "$gflops" ]; then
        printf "%-6s %-14s %s\n" "$n" "(run failed/incomplete)" "$(basename "$latest")"
        continue
    fi

    if [ "$n" -eq 1 ]; then
        base_gflops="$gflops"
    fi

    if [ -n "$base_gflops" ]; then
        speedup=$(awk -v g="$gflops" -v b="$base_gflops" 'BEGIN{printf "%.2fx", g/b}')
        efficiency=$(awk -v g="$gflops" -v b="$base_gflops" -v n="$n" 'BEGIN{printf "%.1f%%", 100*g/(b*n)}')
    else
        speedup="n/a"
        efficiency="n/a"
    fi

    printf "%-6s %-14s %-14s %-10s %-12s %s\n" "$n" "$time_ms" "$gflops" "$speedup" "$efficiency" "$(basename "$latest")"
done
