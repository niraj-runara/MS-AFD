#!/usr/bin/env bash
# M3 (dense) — real Llama-3 8B end to end with the FFN served by the fabric op.
# Builds libmsafd_dense.so, then runs Rung 1 (all-layers correctness) and
# Rungs 2-3 (teacher-forced argmax agreement + greedy generation vs HF).
#
# Requires a Hugging Face token with access to the gated Llama-3 weights, passed
# via the HF_TOKEN environment variable (never hard-coded here).
#
# Usage: HF_TOKEN=hf_xxx bash scripts/run_m3_dense.sh [MODEL] [PROMPT] [MAX_NEW]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

MODEL=${1:-meta-llama/Meta-Llama-3-8B}
PROMPT=${2:-The capital of France is}
MAX_NEW=${3:-20}

if [ -z "${HF_TOKEN:-}" ]; then
    echo "!! HF_TOKEN not set — export a token with access to $MODEL" >&2
    exit 1
fi
# transformers / huggingface_hub read these; keep the token out of argv & logs.
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "===== M3-dense $(date -u) | model=$MODEL ====="
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
cmake --build build -j --target msafd_dense 2>&1 | tail -3
LIB=build/libmsafd_dense.so
[ -f "$LIB" ] || { echo "!! build failed: $LIB missing"; exit 1; }

echo; echo "===== Rung 1 — all-layers correctness ====="
python3 tools/check_all_layers_dense.py --model "$MODEL" --prompt "$PROMPT" --lib "$LIB"

echo; echo "===== Rungs 2-3 — generation vs HF ====="
python3 tools/generate_dense.py --model "$MODEL" --prompt "$PROMPT" \
    --max-new "$MAX_NEW" --lib "$LIB"

echo; echo "===== M3-dense DONE $(date -u) ====="
