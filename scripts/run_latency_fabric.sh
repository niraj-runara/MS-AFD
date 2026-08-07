#!/usr/bin/env bash
# Case 2 latency — real model decode with every MoE FFN on the live fabric.
# Same as run_moe_live.sh but drives tools/latency_bench.py (per-token timing)
# instead of the correctness driver. Launches 48 units + times each decode token.
#
# Usage: bash scripts/run_latency_fabric.sh [MODEL] [WDIR] [PROMPT] [MAX_NEW] [T_MAX]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

MODEL=${1:-Qwen/Qwen3-30B-A3B}
WDIR=${2:-/workspace/m4_weights}
PROMPT=${3:-The capital of France is}
MAX_NEW=${4:-256}
T_MAX=${5:-1024}

[ -f "$WDIR/config.txt" ] || { echo "!! no $WDIR/config.txt — run tools/dump_moe_weights.py first" >&2; exit 1; }
read -r N_LAYERS D F N_EXP TOP_K < "$WDIR/config.txt"
NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
[ "$NGPU" -ge 2 ] || { echo "need >=2 GPUs" >&2; exit 1; }
# GPU0 also hosts HF's A-side (~10 GB), so it gets FEWER units than the others.
# 2-GPU: ~40% of units on GPU0, rest on GPU1. 3+ GPUs: even (GPU0's HF share fits).
if [ "$NGPU" -eq 2 ]; then
    N0=$(( N_LAYERS * 2 / 5 ))                 # units on GPU0
    MAXU=$(( N_LAYERS - N0 ))                  # GPU1 carries the most
else
    N0=-1
    MAXU=$(( (N_LAYERS + NGPU - 1) / NGPU ))
fi
PCT=$(( 100 / MAXU )); [ "$PCT" -lt 1 ] && PCT=1

OUT=results/cmp; mkdir -p "$OUT"
CTRL=/dev/shm/moelive_ctrl; HDIR=/dev/shm/moelive_h
rm -f "$CTRL" "$CTRL.tmp"; mkdir -p "$HDIR"; rm -f "$HDIR"/* 2>/dev/null || true

echo "===== Case2 latency $(date -u) | ${N_LAYERS} units across ${NGPU} GPUs @ ${PCT}% (HF A-side on GPU0) ====="
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p /tmp/mps /tmp/mps_log
echo quit | nvidia-cuda-mps-control 2>/dev/null || true; sleep 1
nvidia-cuda-mps-control -d; sleep 2
echo "start_server -uid $(id -u)" | nvidia-cuda-mps-control 2>/dev/null || true; sleep 2

unit_pids=()
for ((L=0; L<N_LAYERS; L++)); do
    if [ "$N0" -ge 0 ]; then
        if [ "$L" -lt "$N0" ]; then DEV=0; else DEV=1; fi   # 2-GPU skew
    else
        DEV=$(( L % NGPU ))
    fi
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$PCT" \
        ./build/moe_live_unit "$L" "$N_LAYERS" "$D" "$F" "$N_EXP" "$TOP_K" "$T_MAX" \
        "$CTRL" "$HDIR" "$WDIR" "$DEV" "$OUT/fabric_unit_L${L}.csv" \
        > "$OUT/fabric_unit_${L}.log" 2>&1 &
    unit_pids+=($!)
done
echo "launched $N_LAYERS units"

env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 tools/latency_bench.py --mode fabric --model "$MODEL" \
    --lib build/libmsafd_moe_live.so --hdir "$HDIR" --ctrl "$CTRL" --t-max "$T_MAX" \
    --device 0 --prompt "$PROMPT" --max-new "$MAX_NEW" --out "$OUT/fabric.csv" 2>&1 | tee "$OUT/fabric.log"
rc=${PIPESTATUS[0]}

for p in "${unit_pids[@]}"; do wait "$p" 2>/dev/null || true; done
echo "===== Case2 latency DONE rc=$rc ====="
exit "$rc"
