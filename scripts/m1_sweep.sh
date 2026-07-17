#!/usr/bin/env bash
# M1 — determinism vs slice count. Two sweeps:
#   partitioned : each slice gets 100/N% SMs (GPU carved into N micro-units)
#   fixed10     : each slice fixed at 10% SMs while N grows (oversubscription)
#
# Usage: m1_sweep.sh [ITERS] [TOKENS] [ARENA_MB]
set -euo pipefail

ITERS=${1:-2000}
TOKENS=${2:-256}
ARENA_MB=${3:-768}

export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p /tmp/mps /tmp/mps_log
nvidia-cuda-mps-control -d 2>/dev/null || true

BASE=results/m1
mkdir -p "$BASE"
SUMMARY="$BASE/summary.csv"
rm -f "$SUMMARY"

echo "########## M1 sweep: partitioned (100/N% each) ##########"
for N in 1 8 16 32 48; do
    PCT=$((100 / N)); [ "$PCT" -lt 1 ] && PCT=1
    OUT="$BASE/partitioned_N${N}"
    scripts/run_slices.sh "$N" "$PCT" "$ITERS" "$TOKENS" "$OUT" "$ARENA_MB"
    python3 bench/aggregate.py "$OUT" partitioned "$N" "$PCT" "$SUMMARY"
done

echo "########## M1 sweep: fixed 10% each (oversubscribed) ##########"
for N in 8 16 48; do
    PCT=10
    OUT="$BASE/fixed10_N${N}"
    scripts/run_slices.sh "$N" "$PCT" "$ITERS" "$TOKENS" "$OUT" "$ARENA_MB"
    python3 bench/aggregate.py "$OUT" fixed10 "$N" "$PCT" "$SUMMARY"
done

echo; echo "########## M1 summary ($SUMMARY) ##########"
if command -v column >/dev/null 2>&1; then
    column -s, -t "$SUMMARY"
else
    cat "$SUMMARY"
fi
