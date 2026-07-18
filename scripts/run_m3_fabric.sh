#!/usr/bin/env bash
# M3-mini through the fabric — hub + one MPS expert process per active expert on
# a single GPU, real Qwen3 weights via CUDA IPC. Checks correctness vs HF.
#
# Usage: scripts/run_m3_fabric.sh [ref_dir] [beats]
#   NOMPS=1 to skip MPS (experts uncapped; correctness is independent of MPS).
set -euo pipefail

BIN=./build/m3_fabric
if [[ ! -x "$BIN" ]]; then
    echo "build first: cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

REFDIR="${1:-results/m3_ref}"
BEATS="${2:-2000}"
if [[ ! -f "$REFDIR/active_experts.txt" ]]; then
    echo "no $REFDIR/active_experts.txt — run tools/dump_reference.py first" >&2
    exit 1
fi
N=$(grep -c . "$REFDIR/active_experts.txt")
echo "M3-fabric: $N active experts, ref=$REFDIR, beats=$BEATS"

export CUDA_VISIBLE_DEVICES=0
if [[ -z "${NOMPS:-}" ]]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 1
    nvidia-cuda-mps-control -d
    sleep 2
    echo "start_server -uid $(id -u)" | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 2
    echo "MPS mode"
else
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    unset CUDA_MPS_PIPE_DIRECTORY || true
    unset CUDA_MPS_LOG_DIRECTORY || true
    echo "NO-MPS mode"
fi

# Clean stale IPC/control state from a previous run.
rm -f "$REFDIR/input.ipc" "$REFDIR/input.ipc.tmp" \
      "$REFDIR/output.ipc" "$REFDIR/output.ipc.tmp" "$REFDIR/m3_ctrl"

pids=()
"$BIN" hub "$REFDIR" "$BEATS" > "$REFDIR/hub.log" 2>&1 & pids+=($!)
for (( p=0; p<N; p++ )); do
    "$BIN" expert "$p" "$REFDIR" > "$REFDIR/expert_${p}.log" 2>&1 &
done

rc=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then rc=1; fi
done
wait

echo "--- hub ---"
cat "$REFDIR/hub.log"
if (( rc != 0 )); then echo "M3-fabric: hub reported a mismatch/failure — see $REFDIR/*.log" >&2; fi
exit $rc
