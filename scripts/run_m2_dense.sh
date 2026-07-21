#!/usr/bin/env bash
# M2 (dense) — the fabric. A-side (GPU0) scatters token tiles to H F-side GPUs
# (GPU1..GPU H), each running a hub + N MPS micro-units. Needs 1+H GPUs.
# H=1 is the minimal 2-GPU fabric; H>1 exercises the M2N cross-GPU routing.
#
# Usage: scripts/run_m2_dense.sh [nunits] [beats] [pct] [H] [arena_mb]
#   nunits    micro-units per F-side GPU (default 8)
#   beats     timed beats per run       (default 10000; big number = soak)
#   pct       MPS thread% per unit      (default floor(100/nunits))
#   H         number of F-side GPUs     (default = #GPUs - 1)
#   arena_mb  per-unit HBM arena size   (default 768)
# Env: NOMPS=1 skips MPS (units uncapped, default time-slicing).
set -uo pipefail

BIN=./build/m2_dense
if [[ ! -x "$BIN" ]]; then
    echo "build first (needs NCCL): cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
NUNITS="${1:-8}"
BEATS="${2:-10000}"
PCT="${3:-$(( 100 / NUNITS ))}"
H="${4:-$(( NGPU - 1 ))}"
ARENA_MB="${5:-768}"
if (( PCT < 1 )); then PCT=1; fi
if (( H < 1 )); then echo "need >=2 GPUs (found ${NGPU})" >&2; exit 1; fi
if (( H + 1 > NGPU )); then echo "H=${H} needs $((H+1)) GPUs, found ${NGPU}" >&2; exit 1; fi

export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export MSAFD_ARENA_MB="$ARENA_MB"
unset CUDA_VISIBLE_DEVICES || true      # all GPUs visible; each role sets its device

# MPS gives the per-unit SM isolation (the "micro" in micro-unit). Set NOMPS=1 to
# skip it (units contend on the full GPU via default time-slicing) — useful on a
# host whose container can't run the MPS server, and as a determinism contrast.
if [[ -z "${NOMPS:-}" ]]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 1
    nvidia-cuda-mps-control -d
    sleep 2
    echo "MPS mode (units SM-capped at ${PCT}%)"
else
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE || true
    echo "NO-MPS mode (units uncapped, default time-slicing)"
fi

OUT=results/m2_dense
mkdir -p "$OUT"
IDFILE=/dev/shm/m2d_nccl_id
rm -f "$IDFILE" "$IDFILE.tmp"

echo "M2-dense: A-side(GPU0) scatter -> ${H} F-side GPU(s) x ${NUNITS} units @ ${PCT}%, ${BEATS} beats, arena=${ARENA_MB}MB"

# An MPS server allows at most 48 client CUDA contexts. Only the *units* are the
# SM-capped micro-units that need MPS; the A-side and hubs are infrastructure —
# so we run them OUTSIDE MPS (env -u) to keep the whole 48-client budget for
# units (N=48 units => exactly 48 MPS clients). NOMPS mode already has no MPS env.
NOMPS_ENV="env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

pids=()
# A-side (rank 0, GPU0) — infrastructure, not MPS-capped.
$NOMPS_ENV "$BIN" aside "$H" "$NUNITS" "$IDFILE" "$BEATS" "$OUT/aside.csv" \
    > "$OUT/aside.log" 2>&1 & pids+=($!)

# One hub + N units per F-side GPU (GPU g == NCCL rank g, g = 1..H).
for (( g=1; g<=H; g++ )); do
    CTRL="/dev/shm/m2d_ctrl_${g}"
    HDIR="/dev/shm/m2d_handles_${g}"
    rm -f "$CTRL" "$CTRL.tmp"; mkdir -p "$HDIR"; rm -f "$HDIR"/* 2>/dev/null || true
    $NOMPS_ENV "$BIN" hub "$g" "$H" "$NUNITS" "$IDFILE" "$CTRL" "$HDIR" "$BEATS" \
        "$OUT/hub_r${g}.csv" > "$OUT/hub_r${g}.log" 2>&1 & pids+=($!)
    for (( e=0; e<NUNITS; e++ )); do
        # Global unit index across F-GPUs so aggregate.py (globs latency_rank*.csv)
        # reports cross-unit determinism exactly as it does for M1.
        GID=$(( (g-1)*NUNITS + e ))
        UCSV="$OUT/latency_rank${GID}.csv"
        if [[ -z "${NOMPS:-}" ]]; then
            CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$PCT" \
                "$BIN" unit "$e" "$NUNITS" "$CTRL" "$HDIR" "$g" "$BEATS" "$UCSV" \
                > "$OUT/g${g}_unit_${e}.log" 2>&1 &
        else
            "$BIN" unit "$e" "$NUNITS" "$CTRL" "$HDIR" "$g" "$BEATS" "$UCSV" \
                > "$OUT/g${g}_unit_${e}.log" 2>&1 &
        fi
    done
done

rc=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then rc=1; fi
done
wait  # units

echo "--- A-side fabric beat (headline determinism metric) ---"
cat "$OUT/aside.log"
for (( g=1; g<=H; g++ )); do
    echo "--- hub r${g} ---"; cat "$OUT/hub_r${g}.log"
done
if (( rc != 0 )); then echo "M2-dense: a rank failed — check $OUT/*.log" >&2; fi
exit $rc
