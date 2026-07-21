#!/usr/bin/env bash
# M4 (G1) — real Llama-3 8B generation served by the LIVE multi-process fabric.
# Launches one MPS-capped unit process per layer (real weights, IPC), then runs
# HF generation with every FFN dispatched to the fabric. Correctness (vs HF) +
# per-unit decode determinism.
#
# Prereq: weights dumped once via tools/dump_dense_weights.py --out $WDIR
#
# Usage: HF_TOKEN=hf_xxx bash scripts/run_m3_live.sh [MODEL] [WDIR] [PROMPT] [MAX_NEW] [T_MAX] [DEVICE]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

MODEL=${1:-meta-llama/Meta-Llama-3-8B}
WDIR=${2:-/root/m3_weights}
PROMPT=${3:-The capital of France is}
MAX_NEW=${4:-20}
T_MAX=${5:-64}
DEVICE=${6:-0}

[ -f "$WDIR/config.txt" ] || { echo "!! no $WDIR/config.txt — run tools/dump_dense_weights.py first" >&2; exit 1; }
read -r N_LAYERS D F < "$WDIR/config.txt"
PCT=$((100 / N_LAYERS)); [ "$PCT" -lt 1 ] && PCT=1
OUT=results/m4_live; mkdir -p "$OUT"
CTRL=/dev/shm/m3live_ctrl
HDIR=/dev/shm/m3live_h
rm -f "$CTRL" "$CTRL.tmp"; mkdir -p "$HDIR"; rm -f "$HDIR"/* 2>/dev/null || true

echo "===== M4-live $(date -u) | model=$MODEL layers=$N_LAYERS D=$D F=$F pct=$PCT% ====="

# MPS up (units are SM-capped clients; HF stays OUTSIDE MPS).
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p /tmp/mps /tmp/mps_log
echo quit | nvidia-cuda-mps-control 2>/dev/null || true; sleep 1
nvidia-cuda-mps-control -d; sleep 2

# Launch one unit per layer (MPS-capped). They load weights, then wait for the
# coordinator's IPC buffers (created when generate_live.py calls msafd_live_init).
unit_pids=()
for ((L=0; L<N_LAYERS; L++)); do
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$PCT" \
        ./build/m3_live_unit "$L" "$N_LAYERS" "$D" "$F" "$T_MAX" "$CTRL" "$HDIR" "$WDIR" "$DEVICE" \
        "$OUT/latency_rank${L}.csv" > "$OUT/unit_${L}.log" 2>&1 &
    unit_pids+=($!)
done
echo "launched $N_LAYERS units"

# HF coordinator OUTSIDE MPS (env -u) — it is not an SM-capped micro-unit.
env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
    python3 tools/generate_live.py --model "$MODEL" --lib build/libmsafd_live.so \
    --hdir "$HDIR" --ctrl "$CTRL" --t-max "$T_MAX" --device "$DEVICE" \
    --prompt "$PROMPT" --max-new "$MAX_NEW" 2>&1 | tee "$OUT/generate.log"
rc=${PIPESTATUS[0]}

echo "waiting for units to finish..."
for p in "${unit_pids[@]}"; do wait "$p" 2>/dev/null || true; done

echo; echo "===== per-unit decode determinism ====="
python3 bench/aggregate.py "$OUT" m4_live "$N_LAYERS" "$PCT" "$OUT/summary.csv" || true
echo "===== M4-live DONE $(date -u) rc=$rc ====="
exit "$rc"
