#!/usr/bin/env bash
# M2 (dense) autonomous sweep — launch under nohup/tmux so it survives an SSH
# disconnect. Builds, sweeps the micro-unit count over the dense fabric, and
# aggregates two views:
#   - per-unit FFN determinism (M1-comparable) -> results/m2_dense/summary.csv
#   - A-side round-trip beat (whole fabric)    -> results/m2_dense/summary_aside.csv
# Writes results/m2_dense/AUTORUN_DONE when finished (AUTORUN_FAILED on build fail).
#
# Usage: m2_dense_autorun.sh [BEATS] [ARENA_MB] [H]
#   BEATS     timed beats per run   (default 10000)
#   ARENA_MB  per-unit arena size   (default 768)
#   H         F-side GPUs           (default #GPUs - 1)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

BEATS=${1:-10000}
ARENA_MB=${2:-768}
NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
H=${3:-$(( NGPU - 1 ))}
mkdir -p results/m2_dense

echo "===== M2-dense autorun $(date -u) | GPUs=$NGPU H=$H BEATS=$BEATS ARENA=${ARENA_MB}MB ====="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true

# NCCL dev symlink (some images ship only libnccl.so.2; the linker wants .so).
[ -e /usr/lib/x86_64-linux-gnu/libnccl.so ] || \
    ln -sf /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so 2>/dev/null || true

cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
cmake --build build -j 2>&1 | tail -4
if [ ! -x build/m2_dense ]; then
    echo "!!! BUILD FAILED — build/m2_dense missing"
    touch results/m2_dense/AUTORUN_FAILED
    exit 1
fi

SUMMARY=results/m2_dense/summary.csv
ASIDE=results/m2_dense/summary_aside.csv
# Clear any stray per-run artifacts left in the top-level dir (e.g. from a manual
# run) so the first N's `mv latency_rank*.csv` can't sweep them into its aggregate.
rm -f results/m2_dense/latency_rank*.csv results/m2_dense/aside.* \
      results/m2_dense/hub_r*.* results/m2_dense/g*_unit_*.log
rm -f "$SUMMARY" "$ASIDE"
echo "mode,slices,pct_each,rt_p50_ms,rt_p99_ms,rt_ratio" >"$ASIDE"

# Sweep micro-units per F-side GPU. Partitioned share = floor(100/N).
for N in 1 8 16 32 48; do
    PCT=$((100 / N)); [ "$PCT" -lt 1 ] && PCT=1
    RUNDIR=results/m2_dense/partitioned_N${N}
    rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"
    echo "---- M2-dense N=$N pct=$PCT H=$H $(date -u) ----"
    # run_m2_dense writes to results/m2_dense/*; move this run's artifacts aside after.
    timeout 1800 bash scripts/run_m2_dense.sh "$N" "$BEATS" "$PCT" "$H" "$ARENA_MB"
    rc=$?
    # collect this run's CSVs/logs into the per-N dir
    mv results/m2_dense/latency_rank*.csv results/m2_dense/aside.* \
       results/m2_dense/hub_r*.* results/m2_dense/g*_unit_*.log "$RUNDIR"/ 2>/dev/null || true
    if [ $rc -ne 0 ]; then
        echo "!! N=$N failed/timeout rc=$rc — continuing"
        continue
    fi
    python3 bench/aggregate.py "$RUNDIR" m2_dense "$N" "$PCT" "$SUMMARY" || true
    # A-side round-trip beat percentiles from this run's aside.csv
    python3 - "$RUNDIR/aside.csv" "$N" "$PCT" "$ASIDE" <<'PY'
import csv, sys
p, N, PCT, out = sys.argv[1:5]
try:
    ms = sorted(float(r["ms"]) for r in csv.DictReader(open(p)))
except FileNotFoundError:
    sys.exit(0)
def pct(q):
    i = q/100*(len(ms)-1); lo = int(i); f = i-lo
    return ms[-1] if lo+1 >= len(ms) else ms[lo]*(1-f)+ms[lo+1]*f
open(out, "a").write(f"m2_dense,{N},{PCT},{pct(50):.4f},{pct(99):.4f},{pct(99)/pct(50):.4f}\n")
PY
done

echo "===== M2-dense autorun DONE $(date -u) ====="
echo "--- per-unit FFN determinism (summary.csv) ---"; cat "$SUMMARY" 2>/dev/null
echo "--- A-side round-trip beat (summary_aside.csv) ---"; cat "$ASIDE" 2>/dev/null
touch results/m2_dense/AUTORUN_DONE
