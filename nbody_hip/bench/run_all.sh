#!/usr/bin/env bash
# Runs the 1/2/4/5-GPU fp64 nbody benchmarks back-to-back (they must be
# sequential -- they'd otherwise contend for the same GPUs 0..N-1) and
# prints a scaling summary at the end.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for n in 1 2 4 5; do
    "$SCRIPT_DIR/run_bench.sh" "$n" "$@"
    echo
done

"$SCRIPT_DIR/summarize.sh"
