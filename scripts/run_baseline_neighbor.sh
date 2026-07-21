#!/usr/bin/env bash
# G2 (part 2) — the noisy-neighbour control. The N-tenant sweep showed homogeneous
# co-tenancy is already deterministic; the fabric's UNIQUE value is isolation from
# an adversarial neighbour. Here a small latency-sensitive worker runs next to a
# heavy antagonist (big continuous GEMMs), three ways:
#
#   solo   : worker alone                         (ceiling)
#   naive  : worker + heavy antagonist, NO MPS    (worker competes for all SMs)
#   fabric : worker@WPCT% + antagonist capped, MPS(worker owns an isolated slice)
#
# The metric is the WORKER's p50 shift and p99/p50 vs solo — how much the neighbour
# perturbs it. Writes results/baseline_neighbor/.
#
# Usage: scripts/run_baseline_neighbor.sh [ITERS] [WORKER_TOK] [ANTAG_TOK] [N_ANTAG] [WPCT]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

ITERS=${1:-3000}
WTOK=${2:-256}
ATOK=${3:-4096}
NANTAG=${4:-2}
WPCT=${5:-40}
BIN=./build/slice
[ -x "$BIN" ] || { echo "build ./build/slice first" >&2; exit 1; }
OUT=results/baseline_neighbor; mkdir -p "$OUT"
ANTAG_ITERS=100000000

mps_stop() { echo quit | nvidia-cuda-mps-control 2>/dev/null || true; sleep 1; }
mps_start() { mkdir -p /tmp/mps /tmp/mps_log
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
  nvidia-cuda-mps-control -d; sleep 2; }

ANTAG_PIDS=()
kill_antags() { for p in "${ANTAG_PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done; ANTAG_PIDS=(); sleep 1; }
trap kill_antags EXIT
launch_antags() { # $1 = per-antagonist MPS pct ("" = none)
    ANTAG_PIDS=()
    for ((a=0; a<NANTAG; a++)); do
        MSAFD_ARENA_MB=2048 ${1:+CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$1} \
            "$BIN" "$ANTAG_ITERS" "$ATOK" "$OUT/antag_${a}.csv" >/dev/null 2>&1 &
        ANTAG_PIDS+=($!)
    done
    sleep 5
}

mps_stop; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE || true
echo "== solo =="
MSAFD_ARENA_MB=512 "$BIN" "$ITERS" "$WTOK" "$OUT/solo.csv" >"$OUT/solo.log" 2>&1
grep '^\[loop\]' "$OUT/solo.log" || true

echo "== naive (worker + $NANTAG heavy antagonists @${ATOK}tok, NO MPS) =="
launch_antags ""
MSAFD_ARENA_MB=512 "$BIN" "$ITERS" "$WTOK" "$OUT/naive.csv" >"$OUT/naive.log" 2>&1
grep '^\[loop\]' "$OUT/naive.log" || true
kill_antags

APCT=$(( (100 - WPCT) / NANTAG )); [ "$APCT" -lt 1 ] && APCT=1
echo "== fabric (worker@${WPCT}%, $NANTAG antagonists@${APCT}%, MPS) =="
mps_start
launch_antags "$APCT"
MSAFD_ARENA_MB=512 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$WPCT" \
    "$BIN" "$ITERS" "$WTOK" "$OUT/fabric.csv" >"$OUT/fabric.log" 2>&1
grep '^\[loop\]' "$OUT/fabric.log" || true
kill_antags; mps_stop

echo; echo "===== G2 noisy-neighbour — worker FFN under a heavy neighbour ====="
python3 - "$OUT" <<'PY'
import csv, sys, os
out = sys.argv[1]
def stats(name):
    p = os.path.join(out, name + ".csv")
    if not os.path.exists(p): return None
    ms = sorted(float(r["ms"]) for r in csv.DictReader(open(p)))
    def q(x):
        i=x/100*(len(ms)-1); lo=int(i); f=i-lo
        return ms[-1] if lo+1>=len(ms) else ms[lo]*(1-f)+ms[lo+1]*f
    return q(50), q(99), q(99.9), q(99)/q(50)
solo = stats("solo")
rows=[("mode","p50_ms","p99_ms","p99.9_ms","p99/p50","p50 vs solo")]
w=open(os.path.join(out,"summary.csv"),"w"); w.write("mode,p50_ms,p99_ms,p999_ms,ratio,p50_slowdown\n")
for m in ("solo","naive","fabric"):
    s=stats(m)
    if not s: continue
    slow = s[0]/solo[0] if solo else 1.0
    rows.append((m,f"{s[0]:.4f}",f"{s[1]:.4f}",f"{s[2]:.4f}",f"{s[3]:.3f}",f"{slow:.2f}x"))
    w.write(f"{m},{s[0]:.4f},{s[1]:.4f},{s[2]:.4f},{s[3]:.4f},{slow:.4f}\n")
w.close()
wd=[max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
for r in rows: print("  "+"  ".join(c.ljust(wd[i]) for i,c in enumerate(r)))
PY
echo "===== neighbour DONE ====="
