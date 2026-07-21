# MS-AFD — Micro-Sliced Attention–FFN Disaggregation

A software-defined systolic FFN fabric on commodity GPUs: partition GPUs into many
statically-scheduled micro-units that execute FFN layers in a deterministic, systolic
rhythm. See `docs/implementation-plan.docx` for the full plan.

This repo is the **dense-model track** (Llama-3 8B dense FFN). A parallel MoE track
(Qwen3-30B-A3B) lives on the `*-niraj` branches; the two share the same mechanism
and cross-reference each other's findings.

Target GPU: **A100 (Ampere, sm_80)**. Model: **Llama-3 8B dense FFN** — SwiGLU,
`d_model=4096`, `d_intermediate=14336`, fp16 storage / fp32 accumulation.

> Status: **M0, M1, M2 complete.** M2 runs the disaggregated fabric across 2× A100
> over NVLink — A-side scatters token tiles (NCCL) → per-GPU hub → CUDA-IPC fan-out
> to MPS micro-units → gather. At 48 units the whole-fabric beat holds
> **p99/p50 = 1.009**. Full numbers in [results/m2/RESULTS.md](results/m2/RESULTS.md).

## Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** | One slice: MPS + arena + CUDA Graph + persistent loop, measured | ✅ complete (p99/p50=1.001) |
| **M1** | Many slices on one GPU (8→16→48); determinism vs slice count | ✅ complete (p99/p50=1.011 @48) |
| **M2** | 2 GPUs: NCCL scatter/gather + per-GPU hub + CUDA-IPC fan-out to MPS units | ✅ complete (fabric beat p99/p50=1.009 @48) |
| M3 | Real Llama-3 8B end to end; dense FFN on the fabric, correct vs HF | ⬜ next |

Milestones live in this one repo. Tag each as it passes: `m0-complete`, `m1-complete`, …

## Layout

```
src/
  arena.h      # static HBM arena + fixed-offset allocator
  ffn.h        # dense Llama-3 8B FFN (SwiGLU) shape / weights / forward
  slice.cu     # M0/M1: MPS slice — FFN -> CUDA Graph -> persistent loop
  m2_dense.cu  # M2: aside / hub / unit — NCCL scatter + IPC fan-out to MPS units
bench/
  latency.py    # p50 / p99 / p99:p50 from a loop's per-iter timings
  aggregate.py  # per-unit determinism + cross-unit fairness (M1 & M2)
  plot_m1.py    # M1 determinism-vs-slice-count figure
  plot_m2.py    # M2 fabric-determinism-vs-unit-count figure
scripts/
  start_mps.sh          # launch the MPS daemon + client
  run_slices.sh         # M1: N concurrent slices on one GPU
  run_m2_dense.sh       # M2: A-side + hub + N MPS units (1+H GPUs)
  m2_dense_autorun.sh   # M2: unattended sweep -> summary CSVs
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
