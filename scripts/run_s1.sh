#!/usr/bin/env bash
# Spike S1 — launch two NCCL ranks (processes) on ONE GPU under MPS and run the
# graph-captured send/recv test. Pass = both ranks print "S1 PASS".
set -euo pipefail

BIN=./build/s1_graph_nccl
if [[ ! -x "$BIN" ]]; then
    echo "build first (needs NCCL): cmake -S . -B build && cmake --build build -j" >&2
    echo "if the s1 target was skipped, NCCL wasn't found at configure time." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d 2>/dev/null || true

IDFILE=/tmp/s1_nccl_id
rm -f "$IDFILE" "$IDFILE.tmp"

echo "S1: launching rank 0 and rank 1 on GPU 0 (under MPS)..."
"$BIN" 0 "$IDFILE" & p0=$!
"$BIN" 1 "$IDFILE" & p1=$!

rc=0
if ! wait "$p0"; then rc=1; fi
if ! wait "$p1"; then rc=1; fi

if (( rc == 0 )); then
    echo "S1 result: PASS (both ranks OK) — R1 retired, NCCL is graph-capturable"
else
    echo "S1 result: FAIL — see output above; M2 may need NCCL outside the graph" >&2
fi
exit $rc
