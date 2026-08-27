#!/usr/bin/env bash
# Runs the 1/2/4/5/8-GPU fp64 nbody (CUDA) benchmarks back-to-back (they must
# be sequential -- -numdevices=N claims GPUs 0..N-1, so concurrent runs would
# contend for the same devices) and prints a scaling summary at the end.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for n in 1 2 4 5 8; do
    "$SCRIPT_DIR/run_bench.sh" "$n" "$@"
    echo
done

"$SCRIPT_DIR/summarize.sh"
