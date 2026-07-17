#!/usr/bin/env bash
# M2-full — the fabric. A-side (GPU0) scatters to H F-side GPUs (GPU1..GPU H),
# each running a hub + N MPS experts. Needs 1+H GPUs in one instance.
#
# Usage: scripts/run_m2_full.sh [nexp] [beats] [pct] [H]
#   nexp   experts per F-side GPU (default 8)
#   beats  timed beats per run   (default 10000; use a big number for a soak)
#   pct    MPS thread% per expert (default floor(100/nexp))
#   H      number of F-side GPUs (default = (#GPUs - 1))
set -euo pipefail

BIN=./build/m2_full
if [[ ! -x "$BIN" ]]; then
    echo "build first (needs NCCL): cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
NEXP="${1:-8}"
BEATS="${2:-10000}"
PCT="${3:-$(( 100 / NEXP ))}"
H="${4:-$(( NGPU - 1 ))}"
if (( PCT < 1 )); then PCT=1; fi
if (( H < 1 )); then echo "need >=2 GPUs (found ${NGPU})" >&2; exit 1; fi
if (( H + 1 > NGPU )); then echo "H=${H} needs $((H+1)) GPUs, found ${NGPU}" >&2; exit 1; fi

unset CUDA_VISIBLE_DEVICES || true      # all GPUs visible; each role sets device

# MPS is optional. Some container hosts can't run the MPS server (it crashes on
# start). MPS per-expert isolation is already proven (M1/S1.5/M2-vertical); this
# run proves the 4-GPU M2N routing at scale, which doesn't require MPS. Set
# NOMPS=1 to skip MPS (experts share each F-side GPU via default time-slicing,
# uncapped). Leaving CUDA_MPS_PIPE_DIRECTORY unset means clients never try MPS.
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
    echo "MPS mode (experts SM-capped at ${PCT}%)"
else
    # Clear any inherited MPS env (e.g. a manual `export CUDA_MPS_PIPE_DIRECTORY`
    # in the shell) so clients do NOT try to connect to MPS, and stop any stale
    # control daemon left running.
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    unset CUDA_MPS_PIPE_DIRECTORY || true
    unset CUDA_MPS_LOG_DIRECTORY || true
    unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE || true
    echo "NO-MPS mode (experts uncapped, default time-slicing; MPS isolation proven separately)"
fi

IDFILE=/dev/shm/m2f_nccl_id
rm -f "$IDFILE" "$IDFILE.tmp"
mkdir -p results/m2_full

echo "M2-full: A-side(GPU0) scatter -> ${H} F-side GPUs x ${NEXP} experts @ ${PCT}%, ${BEATS} beats"

pids=()
# A-side
"$BIN" aside "$H" "$NEXP" "$IDFILE" "$BEATS" results/m2_full/aside.csv \
    > results/m2_full/aside.log 2>&1 & pids+=($!)

# One hub + N experts per F-side GPU (GPU g == NCCL rank g, g = 1..H).
for (( g=1; g<=H; g++ )); do
    CTRL="/dev/shm/m2f_ctrl_${g}"
    HDIR="/dev/shm/m2f_handles_${g}"
    rm -f "$CTRL" "$CTRL.tmp"; mkdir -p "$HDIR"; rm -f "$HDIR"/*
    "$BIN" hub "$g" "$H" "$NEXP" "$IDFILE" "$CTRL" "$HDIR" "$BEATS" \
        "results/m2_full/hub_r${g}.csv" > "results/m2_full/hub_r${g}.log" 2>&1 & pids+=($!)
    for (( e=0; e<NEXP; e++ )); do
        if [[ -z "${NOMPS:-}" ]]; then
            CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$PCT" \
                "$BIN" expert "$e" "$NEXP" "$CTRL" "$HDIR" "$g" \
                > "results/m2_full/g${g}_expert_${e}.log" 2>&1 &
        else
            "$BIN" expert "$e" "$NEXP" "$CTRL" "$HDIR" "$g" \
                > "results/m2_full/g${g}_expert_${e}.log" 2>&1 &
        fi
    done
done

rc=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then rc=1; fi
done
wait  # experts

echo "--- A-side fabric beat (headline determinism metric) ---"
cat results/m2_full/aside.log
for (( g=1; g<=H; g++ )); do
    echo "--- hub r${g} ---"
    cat "results/m2_full/hub_r${g}.log"
done
if (( rc != 0 )); then echo "M2-full: a rank failed — check results/m2_full/*.log" >&2; fi
exit $rc
