#!/usr/bin/env bash
# Spike S1.5 — intra-GPU IPC fan-out. One hub + N MPS-capped expert processes on
# a single GPU, coordinated each beat via shared memory, data shared via CUDA IPC.
#
# Usage: scripts/run_s15.sh [nexp] [beats] [pct]
#   nexp   expert processes (default 8)
#   beats  timed beats (default 10000)
#   pct    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE per expert (default floor(100/nexp))
set -euo pipefail

BIN=./build/s15_ipc_fanout
if [[ ! -x "$BIN" ]]; then
    echo "build first: cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

NEXP="${1:-8}"
BEATS="${2:-10000}"
PCT="${3:-$(( 100 / NEXP ))}"
if (( PCT < 1 )); then PCT=1; fi

export CUDA_VISIBLE_DEVICES=0           # single GPU (leaves GPU1 free)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d 2>/dev/null || true

CTRL=/dev/shm/s15_ctrl
HDIR=/dev/shm/s15_handles
rm -f "$CTRL" "$CTRL.tmp"
mkdir -p "$HDIR"; rm -f "$HDIR"/*
mkdir -p results/s15

echo "S1.5: hub + ${NEXP} experts on GPU0, ${PCT}% SM each, ${BEATS} beats"
"$BIN" hub "$NEXP" "$CTRL" "$HDIR" "$BEATS" results/s15/beat.csv \
    > results/s15/hub.log 2>&1 & hubpid=$!

for (( e=0; e<NEXP; e++ )); do
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$PCT" \
        "$BIN" expert "$e" "$NEXP" "$CTRL" "$HDIR" \
        > "results/s15/expert_${e}.log" 2>&1 &
done

rc=0
if ! wait "$hubpid"; then rc=1; fi
wait  # let experts exit after stop

echo "--- hub log ---"
cat results/s15/hub.log
if (( rc != 0 )); then
    echo "S1.5: hub failed — check results/s15/*.log" >&2
fi
exit $rc
