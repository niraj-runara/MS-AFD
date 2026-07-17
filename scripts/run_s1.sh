#!/usr/bin/env bash
# Spike S1 — capture ncclSend/ncclRecv + a kernel into one CUDA Graph, replay,
# validate. Retires risk R1. Pass = both ranks print "S1 PASS".
#
# NCCL refuses two ranks on one physical GPU, so:
#   - >=2 GPUs visible: run 1 rank per GPU (cross-GPU, no MPS needed). This is
#     the representative case for M2's routing anyway.
#   - 1 GPU only: falls back to 2-ranks-on-GPU0 under MPS, which NCCL will
#     (expectedly) reject — rent a >=2-GPU box to actually test R1.
set -euo pipefail

BIN=./build/s1_graph_nccl
if [[ ! -x "$BIN" ]]; then
    echo "build first (needs NCCL): cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

IDFILE=/tmp/s1_nccl_id
rm -f "$IDFILE" "$IDFILE.tmp"

NDEV=$(nvidia-smi -L | wc -l | tr -d ' ')
echo "S1: detected ${NDEV} GPU(s)"

if (( NDEV >= 2 )); then
    echo "S1: cross-GPU mode — rank 0 -> GPU 0, rank 1 -> GPU 1"
    unset CUDA_VISIBLE_DEVICES || true
else
    echo "S1: single-GPU mode under MPS (2 ranks share GPU 0; NCCL may reject)"
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d 2>/dev/null || true
fi

"$BIN" 0 "$IDFILE" & p0=$!
"$BIN" 1 "$IDFILE" & p1=$!

rc=0
if ! wait "$p0"; then rc=1; fi
if ! wait "$p1"; then rc=1; fi

if (( rc == 0 )); then
    echo "S1 result: PASS — R1 retired, NCCL is graph-capturable"
else
    echo "S1 result: FAIL — see output above" >&2
fi
exit $rc
