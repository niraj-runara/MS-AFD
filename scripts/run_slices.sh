#!/usr/bin/env bash
# M1 — launch N slice processes concurrently as MPS clients, each capped to
# PCT% of the GPU's SMs, each writing its own per-iteration timing CSV.
#
# Usage: run_slices.sh N PCT ITERS TOKENS OUTDIR [ARENA_MB]
set -euo pipefail

N=${1:?slice count}
PCT=${2:?thread percentage per slice}
ITERS=${3:-2000}
TOKENS=${4:-256}
OUTDIR=${5:?output dir}
ARENA_MB=${6:-768}

export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/mps_log}
export MSAFD_ARENA_MB=$ARENA_MB

mkdir -p "$OUTDIR"
# Fresh barrier dir so all N slices enter their timed loops together.
BARRIER=$(mktemp -d /tmp/msafd_barrier.XXXXXX)
export MSAFD_BARRIER_DIR=$BARRIER
export MSAFD_BARRIER_N=$N
echo "[run_slices] N=$N pct=$PCT iters=$ITERS tokens=$TOKENS arena=${ARENA_MB}MB -> $OUTDIR"

pids=()
for ((i = 0; i < N; i++)); do
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$PCT \
        ./build/slice "$ITERS" "$TOKENS" "$OUTDIR/latency_rank${i}.csv" \
        >"$OUTDIR/rank${i}.log" 2>&1 &
    pids+=("$!")
done

fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=$((fail + 1))
done
rm -rf "$BARRIER"
echo "[run_slices] N=$N done, failures=$fail"
exit "$fail"
