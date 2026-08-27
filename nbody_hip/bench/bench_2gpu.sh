#!/usr/bin/env bash
# 2-GPU fp64 nbody benchmark: nbody_hip_bench -benchmark -numdevices=2 -numbodies=2097152 -fp64
# Logs to bench/logs/nbody_fp64_2gpu_<timestamp>.log. Extra args are passed through.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_bench.sh" 2 "$@"
