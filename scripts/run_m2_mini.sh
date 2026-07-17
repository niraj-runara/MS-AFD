#!/usr/bin/env bash
# M2-mini — one cross-GPU FFN hop. Rank 0 (A-side, GPU 0) drives token tiles to
# rank 1 (F-side expert, GPU 1) and back. Needs >=2 GPUs in one instance.
#
# Usage: scripts/run_m2_mini.sh [beats]
set -euo pipefail

BIN=./build/m2_mini
if [[ ! -x "$BIN" ]]; then
    echo "build first (needs NCCL): cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

BEATS="${1:-10000}"
NDEV=$(nvidia-smi -L | wc -l | tr -d ' ')
if (( NDEV < 2 )); then
    echo "M2-mini needs >=2 GPUs in one instance (found ${NDEV})." >&2
    exit 1
fi

unset CUDA_VISIBLE_DEVICES || true   # both GPUs visible; rank r -> device r
mkdir -p results/m2_mini
IDFILE=/tmp/m2_mini_id
rm -f "$IDFILE" "$IDFILE.tmp"

echo "M2-mini: A-side (GPU0) <-> F-side (GPU1), ${BEATS} beats over NVLink"
"$BIN" 0 "$IDFILE" "$BEATS" results/m2_mini/aside.csv & p0=$!
"$BIN" 1 "$IDFILE" "$BEATS" results/m2_mini/fside.csv & p1=$!

rc=0
if ! wait "$p0"; then rc=1; fi
if ! wait "$p1"; then rc=1; fi

if (( rc == 0 )); then
    echo "--- A-side round-trip beat (the M2 determinism metric) ---"
    python3 bench/latency.py results/m2_mini/aside.csv
    echo "--- F-side hot path (recv -> FFN -> send) ---"
    python3 bench/latency.py results/m2_mini/fside.csv
else
    echo "M2-mini: a rank failed (rc=$rc) — see output above" >&2
fi
exit $rc
