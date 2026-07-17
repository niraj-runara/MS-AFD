#!/usr/bin/env bash
# M1 — launch N slice processes concurrently as MPS clients, each with its own
# SM share and its own CSV, then summarize per-slice determinism.
#
# Usage: scripts/run_fleet.sh N [PCT] [ITERS]
#   N      number of concurrent slices (required)
#   PCT    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE per slice.
#          Default = floor(100/N): equal partition, ~no overcommit
#          (N=8->12, N=16->6, N=48->2). Pass a fixed value to overcommit,
#          e.g. `run_fleet.sh 48 10` => 48 x 10% = 480% demand.
#   ITERS  timed iterations per slice (default 10000)
set -euo pipefail

N="${1:?usage: run_fleet.sh N [PCT] [ITERS]}"
PCT="${2:-$(( 100 / N ))}"
ITERS="${3:-10000}"
if (( PCT < 1 )); then PCT=1; fi

BIN=./build/slice
if [[ ! -x "$BIN" ]]; then
    echo "build first: cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

# MPS env (same as start_mps.sh) + ensure the control daemon is up.
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d 2>/dev/null || true

OUT="results/fleet_N${N}_pct${PCT}"
mkdir -p "$OUT"
echo "M1 fleet: N=$N  pct=${PCT}%  iters=$ITERS  (total demand ~$(( N * PCT ))%)  -> $OUT/"

pids=()
for (( k=0; k<N; k++ )); do
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$PCT" MSAFD_CHECK=0 \
        "$BIN" "$ITERS" "$OUT/slice_${k}.csv" > "$OUT/slice_${k}.log" 2>&1 &
    pids+=("$!")
done

echo "launched ${#pids[@]} slices; waiting..."
fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then fail=$(( fail + 1 )); fi
done
if (( fail )); then
    echo "WARNING: $fail slice(s) exited non-zero — check $OUT/slice_*.log" >&2
fi

echo "--- fleet summary ---"
python3 bench/fleet_summary.py "$OUT"
