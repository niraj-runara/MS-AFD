# MS-AFD — Micro-Sliced Attention–FFN Disaggregation

A software-defined systolic FFN fabric on commodity GPUs: partition GPUs into many
statically-scheduled micro-units that execute FFN layers in a deterministic, systolic
rhythm. See `docs/implementation-plan.docx` for the full plan.

**Current milestone: M0 — single slice.**
Prove one MPS slice, one static arena, one FFN GEMM captured in a CUDA Graph,
running in a persistent loop from HBM with stable (low p99/p50) latency.
The FFN unit = one Qwen3-30B-A3B expert (hidden 2048, ffn 768, bf16, SwiGLU).

> Status: scaffolding only. Source files are skeletons/boilerplate — no implementation yet.

## Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** | One slice: MPS + arena + CUDA Graph + persistent loop, measured | 🚧 scaffolding |
| M1 | Many slices on one GPU (8→16→48); determinism vs slice count | ⬜ later |
| M2 | All 4 F-side GPUs; NCCL M2N routing | ⬜ later |
| M3 | Real A-side + real MoE model, end to end | ⬜ later |

Milestones live in this one repo. Tag each as it passes: `m0-complete`, `m1-complete`, …

## Layout

```
src/
  arena.h     # static 1 GB HBM arena + fixed-offset allocator      (Step 2)
  ffn.h       # FFN GEMM shape / weights                            (Step 3)
  slice.cu    # MPS slice: FFN -> CUDA Graph -> persistent loop      (Steps 1,3,4,5)
bench/
  latency.py  # p50 / p99 / p99:p50 from the loop's per-iter timings (Step 6)
scripts/
  start_mps.sh   # launch the MPS daemon + client                    (Step 1)
docs/
  implementation-plan.docx
```

## Build (planned)

```bash
cmake -S . -B build
cmake --build build
scripts/start_mps.sh          # start MPS, set thread %
./build/slice                 # run the slice
python bench/latency.py       # summarize latency distribution
```

## Requirements

- NVIDIA GPU (A100 80GB, `sm_80`; Hopper/Blackwell also fine), CUDA Toolkit 12.x, cuBLAS
- NCCL (needed from M2; spike S1 tests it earlier)
- Nsight Systems (profiling from day one)
