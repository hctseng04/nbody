#!/usr/bin/env bash
# 8-GPU fp64 nbody (CUDA) benchmark: nbody -benchmark -numdevices=8 -numbodies=2097152 -fp64
# Logs to bench/logs/nbody_fp64_8gpu_<timestamp>.log. Extra args are passed through.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_bench.sh" 8 "$@"
