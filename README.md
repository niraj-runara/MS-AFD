# MS-AFD — Micro-Sliced Attention–FFN Disaggregation

A software-defined systolic FFN fabric on commodity GPUs: partition GPUs into many
statically-scheduled micro-units that execute FFN layers in a deterministic, systolic
rhythm. See `docs/implementation-plan.docx` for the full plan.

**M0 — single slice — ✅ complete.**
One MPS slice, one static arena, one FFN GEMM captured in a CUDA Graph, running
in a persistent loop from HBM with stable (low p99/p50) latency. The FFN unit =
one Qwen3-30B-A3B expert (hidden 2048, ffn 768, bf16, SwiGLU). Random weights at
the expert's *shape* — no real weights are downloaded (those matter only at M3).

**Next: M1 — many slices on one GPU.**

## Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** | One slice: MPS + arena + CUDA Graph + persistent loop, measured | ✅ complete |
| M1 | Many slices on one GPU (8→16→48); determinism vs slice count | 🚧 next |
| M2 | All 4 F-side GPUs; NCCL M2N routing | ⬜ later |
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

## Layout

```
src/
  check.h     # fail-fast CUDA / cuBLAS error macros
  arena.h     # static 64 MB HBM arena + fixed-offset allocator     (Step 2)
  ffn.h       # FFN shape/weights declaration                       (Step 3)
  ffn.cu      # 3 bf16 cuBLAS GEMMs + SwiGLU kernel + fp32 reference (Step 3)
  slice.cu    # MPS slice: FFN -> CUDA Graph -> persistent loop      (Steps 1,3,4,5)
bench/
  latency.py  # p50 / p99 / p99:p50 from the loop's per-iter timings (Step 6)
scripts/
  start_mps.sh   # launch the MPS daemon + set thread %              (Step 1)
docs/
  implementation-plan.docx
```

## Build & run

```bash
cmake -S . -B build && cmake --build build -j
source scripts/start_mps.sh   # NB: source, not ./ — the slice must inherit the
                              # MPS env to connect as a client and be SM-capped
./build/slice                 # optional arg: iteration count (default 10000)
                              #   -> writes latency.csv (one row per launch)
python bench/latency.py       # summarize the distribution (p50/p99/p99:p50)
```

Run `slice` in the same shell you sourced `start_mps.sh` in, or it won't be
MPS-capped. Adjust `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` in `start_mps.sh` to
change the slice's SM share.

## Requirements

- NVIDIA GPU (A100 80GB, `sm_80`; Hopper/Blackwell also fine), CUDA Toolkit 12.x
  or 13.x (tested on 13.0), cuBLAS
- NCCL (needed from M2; spike S1 tests it earlier)
- Nsight Systems (profiling from day one)
