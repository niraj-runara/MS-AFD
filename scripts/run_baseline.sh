#!/usr/bin/env bash
# G2 — determinism baseline / control. Answers "is the fabric MORE deterministic
# than naive co-tenancy?" the honest way: run N timed FFN tenants two ways and
# compare the WORST per-tenant p99/p50 as N rises.
#
#   naive  : N processes, NO MPS  -> the GPU time-slices whole contexts, so each
#            tenant's iterations stall unpredictably behind the others' -> tail grows
#   fabric : N processes, MPS-partitioned at 100/N% each -> every tenant owns a
#            fixed, isolated SM slice -> concurrent, protected -> tail stays flat
#
# Directly comparable to M1 (same slice binary, same barrier). Writes results/baseline/.
#
# Usage: scripts/run_baseline.sh [ITERS] [TOKENS] "[N list]"
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

ITERS=${1:-2000}
TOKENS=${2:-256}
NLIST=${3:-"8 16 32 48"}
BIN=./build/slice
[ -x "$BIN" ] || { echo "build ./build/slice first" >&2; exit 1; }
export MSAFD_ARENA_MB=512
OUT=results/baseline; mkdir -p "$OUT"
SUMMARY="$OUT/summary.csv"; rm -f "$SUMMARY"   # aggregate.py creates it with its own header

mps_stop() { echo quit | nvidia-cuda-mps-control 2>/dev/null || true; sleep 1; }
mps_start() { mkdir -p /tmp/mps /tmp/mps_log
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
  nvidia-cuda-mps-control -d; sleep 2; }

# $1=mode(naive|fabric) $2=N $3=pct("" for none)
run_arm() {
    local mode=$1 N=$2 pct=${3:-}
    local dir="$OUT/${mode}_N${N}"; rm -rf "$dir"; mkdir -p "$dir"
    local bar; bar=$(mktemp -d /tmp/bl_bar.XXXXXX)
    local pids=()
    for ((i=0; i<N; i++)); do
        if [ -n "$pct" ]; then
            MSAFD_BARRIER_DIR="$bar" MSAFD_BARRIER_N="$N" CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$pct" \
                "$BIN" "$ITERS" "$TOKENS" "$dir/latency_rank${i}.csv" >"$dir/r${i}.log" 2>&1 &
        else
            MSAFD_BARRIER_DIR="$bar" MSAFD_BARRIER_N="$N" \
                "$BIN" "$ITERS" "$TOKENS" "$dir/latency_rank${i}.csv" >"$dir/r${i}.log" 2>&1 &
        fi
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    rm -rf "$bar"
    python3 bench/aggregate.py "$dir" "$mode" "$N" "${pct:-0}" "$SUMMARY" >/dev/null 2>&1 || true
}

for N in $NLIST; do
    PCT=$((100 / N)); [ "$PCT" -lt 1 ] && PCT=1
    echo "== N=$N :: naive (no MPS) =="
    mps_stop; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE || true
    run_arm naive "$N" ""
    echo "== N=$N :: fabric (MPS @ ${PCT}%) =="
    mps_start
    run_arm fabric "$N" "$PCT"
    mps_stop
done

echo; echo "===== G2 baseline — worst per-tenant p99/p50: naive vs fabric ====="
column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "===== baseline DONE ====="
