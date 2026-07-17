# MS-AFD — Micro-Sliced Attention–FFN Disaggregation

A software-defined systolic FFN fabric on commodity GPUs: partition GPUs into many
statically-scheduled micro-units that execute FFN layers in a deterministic, systolic
rhythm. See `docs/implementation-plan.docx` for the full plan.

**Current milestone: M0 — single slice.**
Prove one MPS slice, one 1 GB static arena, one FFN GEMM captured in a CUDA Graph,
running in a persistent loop from HBM with stable (low p99/p50) latency.

Target GPU: **A100 (Ampere, sm_80)**. Model: **Llama-3 8B dense FFN** — SwiGLU,
`d_model=4096`, `d_intermediate=14336`, fp16 storage / fp32 accumulation.

> Status: **M0 complete.** Built + run on a 1× A100 80GB (RunPod). Under a 10% MPS
> SM cap: p50 4.493 ms, p99 4.500 ms, **p99/p50 = 1.001** (std 6.5 µs), graph
> output bit-exact vs eager. Full numbers in [results/m0/RESULTS.md](results/m0/RESULTS.md).

## Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** | One slice: MPS + arena + CUDA Graph + persistent loop, measured | ✅ complete (p99/p50=1.001) |
| **M1** | Many slices on one GPU (8→16→48); determinism vs slice count | ✅ complete (p99/p50=1.011 @48) |
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
cmake -S . -B build                    # arch defaults to sm_80 (A100)
cmake --build build
scripts/start_mps.sh                    # start MPS, set thread %
./build/slice [iters] [tokens]         # run the slice (defaults: 5000, 256)
python bench/latency.py                 # summarize latency distribution
```

The slice runs an eager forward, checks it against a CPU reference, captures the
FFN into a CUDA Graph, verifies graph output == eager, then times `iters` graph
launches and writes `latency.csv`.

## Requirements

- NVIDIA A100 (Ampere, sm_80), CUDA Toolkit 12.x, cuBLAS
- NCCL (needed from M2; spike S1 tests it earlier)
- Nsight Systems (profiling from day one)
