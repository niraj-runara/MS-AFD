# MS-AFD — Findings (M0 + M1)

Plain-language record of what the first two milestones actually proved. For the
raw numbers and how to reproduce, see the tables in `../README.md`; the raw
per-iteration CSVs live under `results/`.

All runs are on one rented **A100-SXM4-80GB** (108 SMs), CUDA 13.0. We do **not**
care about latency — the goal is to prove the *mechanism* and *determinism*. The
headline metric is the shape of the per-iteration timing distribution:
**p99/p50** (how much slower the slowest 1% of iterations are than the median).
Lower and flatter = more deterministic.

## The one knob: `pct` (SM share)

`pct` = `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`. NVIDIA MPS uses it to cap what
fraction of the GPU's **SMs** (Streaming Multiprocessors — the compute cores) a
process may use. An A100 has 108 SMs; a slice launched at `pct=10` sees ~10 SMs,
not 108. This is the compute-isolation mechanism (component A) and the "micro" in
micro-unit — `pct` is literally how big each slice is.

In the fleet launcher the per-slice default is `floor(100/N)`, so N slices carve
the GPU into N roughly-equal pieces (N=48 → 2% ≈ 2 SMs each).

## M0 — one deterministic slice

**Question.** Can one process, locked to a slice of the GPU, execute an FFN with a
dead-stable rhythm?

**Setup.** One MPS process: a single static 64 MB HBM arena (one `cudaMalloc`,
fixed offsets), one Qwen3-30B-A3B expert's FFN (SwiGLU, bf16, weights from HBM)
captured into a CUDA Graph, launched in a 10,000-iteration persistent loop.

**Result.**
- Correctness vs an fp32 CPU reference: relative-L2 = 0.00513.
- Unconstrained (108 SMs): p99/p50 = 1.108.
- MPS-capped to 10% (process saw 10 SMs): p99/p50 = 1.109.

**Finding.** The mechanism works and the two pieces don't fight: MPS genuinely
isolates compute, and a static-arena + graph-captured FFN runs deterministically
from HBM. Capping the slice to a sliver of the GPU did **not** degrade its
stability (1.108 → 1.109). This is the atom of the fabric.

## M1 — many deterministic slices on one GPU

**Question (the real one).** If you pack *many* slices onto one GPU, does each
stay deterministic, or do they jitter each other?

**Setup.** N concurrent slice processes, each an independent MPS client at
`floor(100/N)%` SM share (equal partition, ≤96% total demand), each running the
same persistent loop and writing its own CSV. Swept N = 8 → 16 → 48.

**Result.**

| N  | pct | median p50 | worst p99/p50 | p50 spread |
|----|-----|-----------|---------------|------------|
| 8  | 12% | 0.168 ms  | 1.067 | 1.000 |
| 16 | 6%  | 0.304 ms  | 1.078 | 1.003 |
| 48 | 2%  | 1.332 ms  | 1.070 | 1.004 |

**Finding.** Per-slice determinism is **flat** from 8 to 48 slices (~1.07 — even
tighter than the single slice), and slices are treated **fairly** (every slice's
median within 0.4% of the others). No contention cliff, no starvation. You can
subdivide one commodity A100 into **48 independent, deterministic FFN
micro-units** that coexist without stepping on each other. This is the
"software-defined systolic fabric" thesis, demonstrated on hardware nobody
designed for it.

## M1 — overcommit stress test

**Question.** The runs above cleanly *partition* the GPU (shares sum to ≤100%).
Does determinism survive when slices genuinely *fight* over the same SMs?

**Setup.** 48 slices at 10% each = **480% total demand** (~5× oversubscribed).

**Result.**

| 48-slice config | total demand | median p50 | worst p99/p50 | p50 spread |
|-----------------|--------------|-----------|---------------|------------|
| 2% each (partition)   | 96%  | 1.332 ms | 1.070 | 1.004 |
| 10% each (overcommit) | 480% | 0.901 ms | 1.104 | 1.006 |

**Finding.** Even at ~5× oversubscription, determinism barely moved (1.070 →
1.104, still ~1.1) and fairness held (spread 1.006). The fabric degrades
**gracefully** under contention, not catastrophically — not obvious a priori.
p50 is *lower* under overcommit because each slice is allowed ~10 SMs (vs ~2 in
the partition); the tiny FFN doesn't saturate them, so MPS time-multiplexes the
excess demand without adding jitter.

## Spike S1 — is NCCL capturable into a CUDA graph?

**Question.** M2 needs the FFN *and* its NCCL send/recv inside one replayable
CUDA graph. Is that even capture-safe? (Plan's top risk, R1.)

**Setup.** Two processes, one rank each on two GPUs; capture `ncclSend` +
`ncclRecv` + a compute kernel into a single graph, replay it, check the exchanged
data.

**Finding.** Yes — captured, instantiated, replayed, data correct, over NVLink.
R1 retired; M2 does **not** need the "NCCL outside the graph" fallback. A second
finding fell out: **NCCL rejects two ranks on one physical GPU** (even under MPS).
That reshaped M2 — see S1.5.

## Spike S1.5 — intra-GPU fan-out (the bridge)

**Question.** Since NCCL is one-rank-per-GPU, each F-side GPU has a single NCCL
"hub" but many MPS expert micro-units. How does the hub hand tokens to them?

**Setup.** One hub process owns shared input/output device buffers, exported via
**CUDA IPC**; N MPS-capped expert processes open them, run their FFN on their
slice, and signal completion through a small shared-memory control block.

**Finding.** The bridge works and stays deterministic (fan-out beat p99/p50):
N=4 → 1.04, N=8 → 1.20, N=16 → 1.06, N=32 → 1.06, N=48 → 1.10. But it exposed a
**mandatory implementation discipline**: the first N=48 run blew up to 24× until
we switched to `cudaDeviceScheduleBlockingSync` + `sched_yield()`. With CUDA's
default *spinning* sync, ~96 spinning threads across 48 processes starve each
other even on a 128-core box. Blocking sync + yield is required for any
many-process MPS fabric.

## M2 — the fabric

**Question.** Put it all together across GPUs: does an A-side router scattering to
multiple F-side GPUs (each fanning out to its experts) stay deterministic?

**Setup.** Clean 4×A100, NVLink. A-side (GPU0) scatters a token-tile batch to 3
F-side GPUs (`ncclSend`/`ncclRecv`, M2N); each F-side hub receives into an
IPC-shared buffer, fans out to 8 MPS experts, gathers, returns. 24 experts total.

**Result.**

| Config | median p50 | p99/p50 |
|--------|-----------|---------|
| MPS (12% each, concurrent) | 0.696 ms | 1.081 |
| No-MPS (uncapped, contend)  | 2.533 ms | 1.015 |
| **MPS soak (5M beats, ~1 hr)** | 0.700 ms | **1.068** |

**Finding.** The full disaggregated fabric is deterministic across 4 GPUs
(p99/p50 ≈ 1.07 — as tight as a single slice), the 3 F-side GPUs are fair to
within 0.3%, data integrity is 100%, and a ~1-hour / 5-million-beat soak showed
**zero drift** (first-10% p99 0.7494 vs last-10% 0.7441). MPS-capped experts run
3.6× faster per beat than uncapped (concurrency from isolation) — the micro-unit
design paying off. R1–R4 all retired.

An operational lesson worth recording: **the compute was easy; the hosts were
hard.** Three rented boxes were unusable — one with GPUs that fell off the bus,
one shared/oversubscribed (74 GB already in use, `ERR!`), one whose container
couldn't run the MPS server. The rule that emerged: on any new box, the first
command is `nvidia-smi`, and you accept it only if every GPU is idle and
error-free.

## What we have and haven't shown

**Shown (M0–M2, the whole prototype).**
- MPS gives real, fair compute isolation on an A100 (`pct` → SM share).
- A static-arena + CUDA-Graph FFN executes deterministically from HBM.
- Determinism and fairness hold whether one GPU is cleanly partitioned (up to 48
  slices) or 5× oversubscribed.
- NCCL send/recv is CUDA-graph-capturable (S1); a per-GPU hub bridges to local
  MPS experts over CUDA IPC (S1.5).
- **The full fabric — MPS micro-units + static arenas + CUDA-graph FFNs + NCCL
  M2N routing — runs deterministically across 4 GPUs, sustained ~1 hour with no
  drift** (M2). Every top risk R1–R4 retired.

**Not yet shown (out of scope for the prototype).**
- We measure **rhythm stability**, not speed. Latency grows as slices shrink; we
  don't care — the ratio is the metric.
- Random weights at the expert's *shape* — no real model weights, no real
  attention/router. A real MoE model end to end is **M3**.
- "Hundreds" of experts across the full fabric simultaneously: the mechanism is
  proven and scales cleanly, but the largest runs so far are 48 experts/GPU
  (S1.5) and 24 across the 3-GPU fabric (M2) — "hundreds live" is extrapolation.
- **R2 (Tensor-Core utilization at low thread %)** — never measured. Determinism
  is proven; whether a 2–12%-SM expert keeps the Tensor Cores *busy* (efficiency)
  is open telemetry, not a determinism gate.
- The L2/SRAM weight-pinning idea was dropped early: the physics don't work
  (~1 MB L2 per slice vs tens–hundreds of MB of weights). Weights are served from
  HBM. The goal was never speed — it was determinism, and that held.
