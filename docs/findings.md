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

## What we have and haven't shown

**Shown.**
- MPS gives real, fair compute isolation on an A100 (`pct` → SM share).
- A static-arena + CUDA-Graph FFN executes deterministically from HBM.
- Determinism and fairness hold whether the GPU is cleanly partitioned (up to 48
  slices) or 5× oversubscribed.

**Not yet shown (deliberately out of scope for M0/M1).**
- We measure **rhythm stability**, not speed. Latency grows as slices shrink
  (0.14 ms → 1.33 ms); we don't care.
- Random weights at the expert's *shape* — no real model weights (those matter
  only at M3).
- Single GPU only. Cross-GPU routing (A-side attention/KV ↔ F-side FFN over
  NCCL) is M2.
- The L2/SRAM weight-pinning idea was dropped early: the physics don't work
  (~1 MB L2 per slice vs tens–hundreds of MB of weights). Weights are served from
  HBM. The goal was never speed — it was determinism, and that held.
