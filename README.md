# MS-AFD — Micro-Sliced Attention–FFN Disaggregation

A software-defined systolic FFN fabric on commodity GPUs: partition GPUs into many
statically-scheduled micro-units that execute FFN layers in a deterministic, systolic
rhythm. See `docs/implementation-plan.docx` for the full plan.

**M0 — single slice — ✅ complete.**
One MPS slice, one static arena, one FFN GEMM captured in a CUDA Graph, running
in a persistent loop from HBM with stable (low p99/p50) latency. The FFN unit =
one Qwen3-30B-A3B expert (hidden 2048, ffn 768, bf16, SwiGLU). Random weights at
the expert's *shape* — no real weights are downloaded (those matter only at M3).

**M1 — many slices on one GPU — ✅ complete.** 8 → 16 → 48 concurrent MPS-isolated
slices on one A100; per-slice determinism stays flat (~1.07) and fairness is
near-perfect as slice count grows.

**Next: M2 — all F-side GPUs; NCCL M2N routing.**

## Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** | One slice: MPS + arena + CUDA Graph + persistent loop, measured | ✅ complete |
| **M1** | Many slices on one GPU (8→16→48); determinism vs slice count | ✅ complete |
| M2 | All 4 F-side GPUs; NCCL M2N routing | 🚧 next |
| M3 | Real A-side + real MoE model, end to end | ⬜ later |

Milestones live in this one repo. Tag each as it passes: `m0-complete`, `m1-complete`, …

## M0 results

Rented A100-SXM4-80GB (108 SMs), CUDA 13.0. 10,000-iteration persistent loop.
Correctness gate: relative-L2 vs an fp32 CPU reference = **0.00513** (≤ 0.03).

| Run | SMs visible | p50 (ms) | p99 (ms) | **p99/p50** |
|-----|-------------|----------|----------|-------------|
| Bare (no MPS) | 108 | 0.0379 | 0.0420 | 1.108 |
| **MPS 10%** | **10** | 0.1403 | 0.1556 | **1.109** |

Takeaways: MPS genuinely partitions the GPU (the process sees 10 SMs, not 108),
and that isolation does **not** degrade determinism — p99/p50 is essentially
identical to the unconstrained baseline. Latency scales with the SM share; we
don't care about latency, only the ratio. Raw per-iteration timings are the CSVs
written by `slice` (one row per launch).

## M1 results

Same A100. N concurrent slices, each an MPS client at `floor(100/N)%` SM share
(equal partition, ~96% total demand, no overcommit), 10,000 iterations each.

| N slices | SM % each | median p50 (ms) | **worst p99/p50** | p50 spread (max/min) |
|----------|-----------|-----------------|-------------------|----------------------|
| 1 (M0)   | 10% | 0.140 | 1.109 | — |
| 8        | 12% | 0.168 | **1.067** | 1.000 |
| 16       | 6%  | 0.304 | **1.078** | 1.003 |
| 48       | 2%  | 1.332 | **1.070** | 1.004 |

Takeaways: per-slice determinism is **flat** from 8 to 48 slices (~1.07, tighter
than the single slice) and fairness is near-perfect (every slice within 0.4% of
the others). No contention cliff, no starvation — 48 deterministic FFN
micro-units co-reside on one A100. p50 scales with the inverse SM share (we don't
care — the ratio is the metric).

## Layout

```
src/
  check.h     # fail-fast CUDA / cuBLAS error macros
  arena.h     # static 64 MB HBM arena + fixed-offset allocator     (Step 2)
  ffn.h       # FFN shape/weights declaration                       (Step 3)
  ffn.cu      # 3 bf16 cuBLAS GEMMs + SwiGLU kernel + fp32 reference (Step 3)
  slice.cu    # MPS slice: FFN -> CUDA Graph -> persistent loop      (Steps 1,3,4,5)
bench/
  latency.py       # p50/p99/p99:p50 from one loop's per-iter timings (Step 6)
  fleet_summary.py # M1: per-slice determinism + cross-slice fairness
scripts/
  start_mps.sh   # launch the MPS daemon + set thread %              (Step 1)
  run_fleet.sh   # M1: launch N concurrent slices, summarize the fleet
docs/
  implementation-plan.docx
```

## Build & run

```bash
cmake -S . -B build && cmake --build build -j
source scripts/start_mps.sh   # NB: source, not ./ — the slice must inherit the
                              # MPS env to connect as a client and be SM-capped
./build/slice                 # optional args: [iters] [out.csv] (default 10000, latency.csv)
                              #   -> writes the CSV (one row per launch)
python bench/latency.py       # summarize the distribution (p50/p99/p99:p50)
```

**M1 — many slices on one GPU.** `run_fleet.sh` spawns N concurrent slices, each
its own MPS client at `floor(100/N)%` SM share, then summarizes the fleet:

```bash
bash scripts/run_fleet.sh 8         # N=8, equal partition (12% each)
bash scripts/run_fleet.sh 16        # N=16 (6% each)
bash scripts/run_fleet.sh 48        # N=48 (2% each)
# override SM% to stress under overcommit: run_fleet.sh 48 10  -> 48 x 10% = 480%
# CSVs land in results/fleet_N<N>_pct<PCT>/ ; summary printed at the end.
```

Run `slice` in the same shell you sourced `start_mps.sh` in, or it won't be
MPS-capped. Adjust `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` in `start_mps.sh` to
change the slice's SM share.

## Requirements

- NVIDIA GPU (A100 80GB, `sm_80`; Hopper/Blackwell also fine), CUDA Toolkit 12.x
  or 13.x (tested on 13.0), cuBLAS
- NCCL (needed from M2; spike S1 tests it earlier)
- Nsight Systems (profiling from day one)
