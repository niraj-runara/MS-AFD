#!/usr/bin/env bash
# M4 — real Qwen3-30B-A3B generation served by the LIVE multi-process MoE fabric.
# HF (A-side) runs on GPU 0, OUTSIDE MPS. One MPS-capped unit process per layer,
# holding that layer's real experts, spread across the F-side GPUs (1..NGPU-1),
# each serving its MoE FFN over CUDA IPC during live generation.
#
# Prereq: weights dumped once via tools/dump_moe_weights.py --out $WDIR
# Usage: bash scripts/run_moe_live.sh [MODEL] [WDIR] [PROMPT] [MAX_NEW] [T_MAX]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

MODEL=${1:-Qwen/Qwen3-30B-A3B}
WDIR=${2:-/workspace/m4_weights}
PROMPT=${3:-The capital of France is}
MAX_NEW=${4:-20}
T_MAX=${5:-1024}

[ -f "$WDIR/config.txt" ] || { echo "!! no $WDIR/config.txt — run tools/dump_moe_weights.py first" >&2; exit 1; }
read -r N_LAYERS D F N_EXP TOP_K < "$WDIR/config.txt"

NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
FSIDE=$((NGPU - 1))
[ "$FSIDE" -ge 1 ] || { echo "need >=2 GPUs (found $NGPU)" >&2; exit 1; }
# units per F-side GPU (ceil), and MPS % per unit on a GPU
UPG=$(( (N_LAYERS + FSIDE - 1) / FSIDE ))
PCT=$(( 100 / UPG )); [ "$PCT" -lt 1 ] && PCT=1

OUT=results/m4; mkdir -p "$OUT"
CTRL=/dev/shm/moelive_ctrl
HDIR=/dev/shm/moelive_h
rm -f "$CTRL" "$CTRL.tmp"; mkdir -p "$HDIR"; rm -f "$HDIR"/* 2>/dev/null || true

echo "===== M4-live $(date -u) | model=$MODEL layers=$N_LAYERS D=$D F=$F experts=$N_EXP top_k=$TOP_K ====="
echo "GPUs=$NGPU (HF on GPU0, units on GPU1..$FSIDE), ~$UPG units/GPU @ ${PCT}% MPS each"

# MPS up (units are SM-capped clients; HF stays OUTSIDE MPS).
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p /tmp/mps /tmp/mps_log
echo quit | nvidia-cuda-mps-control 2>/dev/null || true; sleep 1
nvidia-cuda-mps-control -d; sleep 2
echo "start_server -uid $(id -u)" | nvidia-cuda-mps-control 2>/dev/null || true; sleep 2

# One unit per layer, round-robin over F-side GPUs (device = 1 + L%FSIDE).
unit_pids=()
for ((L=0; L<N_LAYERS; L++)); do
    DEV=$(( 1 + (L % FSIDE) ))
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$PCT" \
        ./build/moe_live_unit "$L" "$N_LAYERS" "$D" "$F" "$N_EXP" "$TOP_K" "$T_MAX" \
        "$CTRL" "$HDIR" "$WDIR" "$DEV" "$OUT/latency_L${L}.csv" \
        > "$OUT/unit_${L}.log" 2>&1 &
    unit_pids+=($!)
done
echo "launched $N_LAYERS units"

# HF coordinator OUTSIDE MPS (env -u) — it is not an SM-capped micro-unit.
env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
    python3 tools/generate_moe_live.py --model "$MODEL" --lib build/libmsafd_moe_live.so \
    --hdir "$HDIR" --ctrl "$CTRL" --t-max "$T_MAX" --device 0 \
    --prompt "$PROMPT" --max-new "$MAX_NEW" 2>&1 | tee "$OUT/generate.log"
rc=${PIPESTATUS[0]}

echo "waiting for units to finish..."
for p in "${unit_pids[@]}"; do wait "$p" 2>/dev/null || true; done
echo; echo "===== per-unit decode determinism ====="
python3 bench/latency.py "$OUT"/latency_L0.csv 2>/dev/null || true
echo "(per-unit CSVs in $OUT/latency_L*.csv)"
echo "===== M4-live DONE $(date -u) rc=$rc ====="
exit "$rc"
