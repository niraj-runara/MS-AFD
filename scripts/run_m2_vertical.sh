#!/usr/bin/env bash
# M2-vertical — A-side (GPU0) <-NCCL-> F-side hub (GPU1) <-IPC-> N MPS experts.
# Needs >=2 GPUs in one instance. Uses GPU0 for the A-side, GPU1 for the F-side.
#
# Usage: scripts/run_m2_vertical.sh [nexp] [beats] [pct]
set -euo pipefail

BIN=./build/m2_vertical
if [[ ! -x "$BIN" ]]; then
    echo "build first (needs NCCL): cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

NEXP="${1:-8}"
BEATS="${2:-10000}"
PCT="${3:-$(( 100 / NEXP ))}"
if (( PCT < 1 )); then PCT=1; fi

NDEV=$(nvidia-smi -L | wc -l | tr -d ' ')
if (( NDEV < 2 )); then
    echo "M2-vertical needs >=2 GPUs in one instance (found ${NDEV})." >&2
    exit 1
fi

unset CUDA_VISIBLE_DEVICES || true      # both GPUs visible; roles pick device
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
# Restart MPS so the daemon exposes BOTH GPUs. A daemon left over from a
# single-GPU run (CUDA_VISIBLE_DEVICES=0) would hide GPU1 from all clients ->
# cudaErrorInvalidDevice on cudaSetDevice(1). A plain `-d` won't fix a stale one.
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
sleep 1
nvidia-cuda-mps-control -d
sleep 1

CTRL=/dev/shm/m2v_ctrl
HDIR=/dev/shm/m2v_handles
IDFILE=/dev/shm/m2v_nccl_id
rm -f "$CTRL" "$CTRL.tmp" "$IDFILE" "$IDFILE.tmp"
mkdir -p "$HDIR"; rm -f "$HDIR"/*
mkdir -p results/m2_vertical

echo "M2-vertical: A-side(GPU0) <-> hub(GPU1) + ${NEXP} experts @ ${PCT}%, ${BEATS} beats"
"$BIN" hub   "$NEXP" "$IDFILE" "$CTRL" "$HDIR" "$BEATS" results/m2_vertical/hub.csv \
    > results/m2_vertical/hub.log 2>&1 & hubpid=$!
"$BIN" aside "$NEXP" "$IDFILE" "$BEATS" results/m2_vertical/aside.csv \
    > results/m2_vertical/aside.log 2>&1 & apid=$!
for (( e=0; e<NEXP; e++ )); do
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$PCT" \
        "$BIN" expert "$e" "$NEXP" "$CTRL" "$HDIR" \
        > "results/m2_vertical/expert_${e}.log" 2>&1 &
done

rc=0
if ! wait "$apid";  then rc=1; fi
if ! wait "$hubpid"; then rc=1; fi
wait

echo "--- aside (round-trip beat = full routed+fanned pipeline) ---"
cat results/m2_vertical/aside.log
echo "--- hub (recv + fan-out + send) ---"
cat results/m2_vertical/hub.log
if (( rc != 0 )); then echo "M2-vertical: a rank failed — check results/m2_vertical/*.log" >&2; fi
exit $rc
