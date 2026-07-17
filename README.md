# MS-AFD — Micro-Sliced Attention–FFN Disaggregation

A software-defined systolic FFN fabric on commodity GPUs: partition GPUs into many
statically-scheduled micro-units that execute FFN layers in a deterministic, systolic
rhythm. See `docs/implementation-plan.docx` for the full plan and
[`docs/findings.md`](docs/findings.md) for what M0–M2 proved (plain-language).

**The prototype is complete (M0 → M2).** Software-defined, deterministic, systolic
FFN execution across commodity GPUs — MPS micro-units + static HBM arenas +
CUDA-graph FFNs + NCCL M2N routing — demonstrated end to end on 4 GPUs, sustained
for an hour with no drift. Every top risk (R1–R4) retired.

**M0 — single slice — ✅.** One MPS slice, one static arena, one FFN GEMM in a
CUDA Graph, persistent loop from HBM, stable p99/p50. FFN unit = one
Qwen3-30B-A3B expert (hidden 2048, ffn 768, bf16, SwiGLU); random weights at the
expert's *shape* — no real weights (those matter only at M3).

**M1 — many slices on one GPU — ✅.** 8→16→48 MPS-isolated slices on one A100;
per-slice determinism flat (~1.07), fairness near-perfect, holds under 5× overcommit.

**M2 — the fabric — ✅.** A-side router scatters token tiles across 3 F-side GPUs
(NCCL M2N) → each GPU fans out to its MPS expert micro-units (CUDA IPC) → results
gather back. p99/p50 = **1.068**, no drift over a 5M-beat / ~1-hour soak.

Getting there required two spikes: **S1** (NCCL send/recv is CUDA-graph-capturable)
and **S1.5** (intra-GPU IPC fan-out — because NCCL allows only one rank per GPU, so
each GPU's rank bridges to its local experts over CUDA IPC).

**Next: M3 — real MoE model + real attention A-side, end to end.**

## Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** | One slice: MPS + arena + CUDA Graph + persistent loop, measured | ✅ complete |
| **M1** | Many slices on one GPU (8→16→48); determinism vs slice count | ✅ complete |
| **M2** | 4 GPUs; NCCL M2N routing + intra-GPU IPC fan-out; 1-hr soak | ✅ complete |
| M3 | Real A-side + real MoE model, end to end | ⬜ next |

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

**Overcommit stress test.** Same 48 slices but at 10% SM share each = **480%
total demand** (~5× oversubscribed), vs the 96% clean partition above:

| 48-slice config | total demand | median p50 (ms) | worst p99/p50 | p50 spread |
|-----------------|--------------|-----------------|---------------|------------|
| 2% each (partition)  | 96%  | 1.332 | 1.070 | 1.004 |
| 10% each (overcommit) | 480% | 0.901 | 1.104 | 1.006 |

Even at ~5× oversubscription, determinism holds (worst p99/p50 1.10, still ~1.1)
and fairness is intact (spread 1.006) — the fabric degrades **gracefully** under
contention, not catastrophically. p50 is *lower* under overcommit because each
slice is allowed ~10 SMs (vs ~2 in the partition); the tiny FFN doesn't saturate
them, so MPS time-multiplexes the excess demand without adding jitter.

## M2 results — the fabric

Clean 4×A100 (NV12 NVLink between all pairs), CUDA 13.0. A-side (GPU0) scatters a
token-tile batch to 3 F-side GPUs (`ncclSend`/`ncclRecv`, M2N); each F-side GPU's
hub receives into a CUDA-IPC-shared buffer, signals its 8 MPS expert processes,
gathers their FFN outputs, and returns them. 24 experts total. 100% data integrity
(every element round-trips).

| Config | median p50 (ms) | p99/p50 | notes |
|--------|-----------------|---------|-------|
| MPS (8 experts/GPU @ 12%) | 0.696 | **1.081** | 10k beats; experts SM-capped, run concurrently |
| No-MPS (uncapped) | 2.533 | 1.015 | 10k beats; experts contend for the full GPU |
| **MPS soak** | **0.700** | **1.068** | **5,000,000 beats (~1 hr), no drift** |

Takeaways:
- **The full disaggregated fabric is deterministic across 4 GPUs** — p99/p50 ≈ 1.07,
  as tight as a single slice. Cross-GPU NCCL routing + intra-GPU IPC fan-out did
  not degrade the rhythm.
- **Perfect cross-GPU fairness** — the 3 F-side GPUs' beats land within 0.3% of
  each other; no straggler.
- **Sustained & stable** — over a 5M-beat / ~1-hour soak, the first-10% and
  last-10% percentiles are identical (p99 0.7494 → 0.7441): **zero drift**.
- **MPS isolation pays off in throughput** — with SM-capped experts running
  concurrently, per-beat time is 3.6× lower than the uncapped no-MPS mode (which
  we don't care about for determinism, but it confirms the micro-unit design).

Two enabling findings behind the fabric (see `docs/findings.md`): **NCCL allows
only one rank per GPU**, so routing is at GPU granularity with a per-GPU hub that
bridges to local experts over **CUDA IPC**; and a **blocking-sync + `sched_yield`**
discipline is mandatory, or a many-process MPS fleet starves itself on spin
contention.

## Layout

```
src/
  check.h        # fail-fast CUDA / cuBLAS error macros
  arena.h        # static 64 MB HBM arena + fixed-offset allocator
  ffn.h / ffn.cu # FFN: 3 bf16 cuBLAS GEMMs + SwiGLU kernel + fp32 reference
  slice.cu       # M0/M1: MPS slice — FFN -> CUDA Graph -> persistent loop
  m2_mini.cu     # M2: one cross-GPU FFN hop (A-side <-NCCL-> F-side expert)
  m2_vertical.cu # M2: full F-side stack on 1 GPU (NCCL hub + IPC fan-out + MPS)
  m2_full.cu     # M2: the fabric — A-side scatters to H F-side GPUs (M2N)
spikes/
  s1_graph_nccl.cu  # S1: NCCL send/recv captured into a CUDA graph
  s15_ipc_fanout.cu # S1.5: hub -> N MPS experts via CUDA IPC (intra-GPU fan-out)
bench/
  latency.py       # p50/p99/p99:p50 from one loop's per-iter timings
  fleet_summary.py # M1: per-slice determinism + cross-slice fairness
scripts/
  start_mps.sh      # launch the MPS daemon + set thread %
  run_fleet.sh      # M1: N concurrent slices, summarize the fleet
  run_s1.sh         # S1 spike (2 GPUs, 1 rank each)
  run_s15.sh        # S1.5 spike (hub + N experts, 1 GPU)
  run_m2_mini.sh    # M2 single hop (2 GPUs)
  run_m2_vertical.sh# M2 F-side stack (2 GPUs)
  run_m2_full.sh    # M2 fabric (1+H GPUs); NOMPS=1 to skip MPS
docs/
  implementation-plan.docx
  findings.md       # plain-language: what M0–M2 proved
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

**M2 — the fabric** (needs 1+H GPUs in one instance; H F-side GPUs). The spikes
first, then the fabric:

```bash
bash scripts/run_s1.sh              # S1: NCCL graph-capture (2 GPUs)
bash scripts/run_s15.sh 8           # S1.5: intra-GPU IPC fan-out (1 GPU)
bash scripts/run_m2_full.sh 8       # the fabric, with MPS (H = #GPUs-1)
NOMPS=1 bash scripts/run_m2_full.sh 8            # if the host can't run MPS
bash scripts/run_m2_full.sh 8 5000000            # ~1-hour soak
```

> **Host hygiene (learned the hard way).** The first command on any rented box
> must be `nvidia-smi` — accept it only if every GPU is idle (~0 MiB, 0% util,
> no `ERR!`); a loaded/`ERR!` box is shared or broken, destroy it. For multi-GPU
> runs, verify NVLink with `nvidia-smi topo -m` (expect `NV#` between pairs). Some
> container hosts can't run the MPS server (server crashes on start) — use
> `NOMPS=1` there. Failed multi-process runs can orphan GPU-holding processes;
> `pkill -9 -f m2_full` (etc.) before retrying.

## Requirements

- NVIDIA GPU (A100 80GB, `sm_80`; Hopper/Blackwell also fine), CUDA Toolkit 12.x
  or 13.x (tested on 13.0), cuBLAS
- NCCL (M2 / spike S1). M2 needs a multi-GPU, NVLink-connected instance
- Nsight Systems / Compute (optional; R2 Tensor-Core telemetry)
