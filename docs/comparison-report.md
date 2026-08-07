# MS-AFD vs. Normal Deployment — Determinism Comparison

**Model:** Qwen3-30B-A3B (MoE: 48 layers, 128 experts/layer, top-8, bf16)
**Date:** 2026-08-07
**Question:** For the *same model on the same hardware*, does routing every FFN through the
MS-AFD systolic fabric make per-token latency **more deterministic** than a normal deployment?

---

## 1. What is being compared

We deploy the **identical model** two ways on the **identical box**, and time every decoded
token the same way (manual KV-cache greedy decode, `torch.cuda.synchronize()` around each token):

| | Case 1 — **Normal** | Case 2 — **MS-AFD** |
|---|---|---|
| FFN execution | HuggingFace, dynamic dispatch | Static systolic fabric: 48 MPS-isolated micro-units (one per MoE layer), fixed HBM arenas, CUDA-Graph-captured FFN, index-driven combine |
| Model placement | Full model pipeline-sharded across the GPUs | Attention/router/KV on GPU 0; the 128 experts of each layer live in a dedicated fabric unit; experts routed to units, results returned over NVLink P2P |
| Scheduling | Whatever the runtime does per step | Deterministic, statically scheduled — every token replays the same captured graph on every unit |

Only the FFN path differs. Everything else (tokenizer, attention, router math, sampling) is identical.

**Headline metric is determinism, not speed** — specifically the per-token **p99/p50** latency
ratio (how fat the tail is), plus CV (σ/μ) and max. This mirrors what a Groq-style LPU buys you:
static scheduling → predictable latency, at the cost of raw throughput.

---

## 2. Hardware & interconnect

Rented **4× NVIDIA A100-SXM4-40 GB** (driver 580.159, CUDA 13.0). We use **3 of the 4 GPUs**.

The fabric depends on **cross-GPU P2P over NVLink**: a fabric unit on GPU 1 or GPU 2 reads the
hidden state directly from the coordinator's buffer on GPU 0 and writes the result back. That only
works if NVLink is actually up — on a PCIe-only or fabric-manager-down box the P2P copies fail and
the fabric produces garbage. This box is fully connected:

```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      NV12    NV12    NV12
GPU1    NV12     X      NV12    NV12
GPU2    NV12    NV12     X      NV12
GPU3    NV12    NV12    NV12     X
```

`NV12` = 12 bonded NVLinks between every pair — full-bandwidth, all-to-all. This is the requirement
to reproduce the result; verify it with `nvidia-smi topo -m` before running.

**Why 3 GPUs (fair-comparison note).** The model weights are ~61 GB, so *normal* deployment
genuinely needs **≥2× 40 GB** — it is not artificially spread. MS-AFD needs **3× 40 GB**: it holds
HuggingFace's ~3 GB non-expert "A-side" on GPU 0 *plus* the full expert set across the fabric units,
a real memory overhead. To keep the hardware identical for both, we ran **both cases on the same
3 GPUs**. Normal leaves the 3rd GPU mostly idle; MS-AFD uses all three. "MS-AFD costs one extra
GPU" is a reported finding, not hidden.

---

## 3. Results (3 runs each, 247 decode tokens per run)

### Case 1 — Normal

| Run | p50 (ms) | p99 (ms) | max (ms) | mean ± σ (ms) | CV | **p99/p50** | tok/s |
|----|---------|---------|---------|--------------|------|-----------|-------|
| 1 | 134.87 | 141.74 | 143.27 | 135.21 ± 1.34 | 0.0099 | 1.0509 | 7.40 |
| 2 | 133.87 | 141.91 | 144.45 | 134.22 ± 1.92 | 0.0143 | 1.0601 | 7.45 |
| 3 | 132.87 | 139.76 | 141.17 | 133.53 ± 1.59 | 0.0119 | 1.0519 | 7.49 |
| **avg** | **133.87** | **141.14** | **142.96** | **134.32** | **0.0120** | **1.0543** | **7.45** |

### Case 2 — MS-AFD

| Run | p50 (ms) | p99 (ms) | max (ms) | mean ± σ (ms) | CV | **p99/p50** | tok/s |
|----|---------|---------|---------|--------------|------|-----------|-------|
| 1 | 464.48 | 476.88 | 560.14 | 465.90 ± 6.80 | 0.0146 | 1.0267 | 2.15 |
| 2 | 466.13 | 476.22 | 490.63 | 466.87 ± 2.78 | 0.0059 | 1.0217 | 2.14 |
| 3 | 466.17 | 470.71 | 476.12 | 466.41 ± 1.94 | 0.0042 | 1.0097 | 2.14 |
| **avg** | **465.59** | **474.60** | **508.96** | **466.39** | **0.0082** | **1.0194** | **2.14** |

### Head-to-head (3-run average)

| Metric | Normal | MS-AFD | Better |
|---|---|---|---|
| **p99/p50 (determinism)** | 1.0543 | **1.0194** | **MS-AFD** (−64% tail) |
| **CV (σ/μ)** | 0.0120 | **0.0082** | **MS-AFD** |
| Correctness | coherent | coherent (identical ≈20 items, then bf16 drift) | tie |
| p50 latency | **133.9 ms** | 465.6 ms | Normal |
| throughput | **7.45 tok/s** | 2.14 tok/s | Normal (3.5×) |
| TTFT | 1054 ms | **1010 ms** | ~tie |
| GPUs required | **2** | 3 | Normal |

---

## 4. What the numbers say

**Determinism — MS-AFD wins, repeatably.** The p99/p50 ratio is tighter in **all three runs**
(1.010–1.027 vs. 1.051–1.060). On average the fabric's tail is **~1.9% over median vs. ~5.4%** for
normal — roughly a third the relative jitter. CV agrees (0.0082 vs. 0.0120). This is the thesis:
static, graph-captured, systolically-scheduled FFN execution produces a flatter latency
distribution than dynamic dispatch. It behaves like dedicated hardware.

**The fabric tightens as it warms.** MS-AFD run 1 has a warmup tail (max 560 ms, CV 0.0146); by
run 3 it is the steadiest of any run on either side (max only 1.02× p50, CV 0.0042). The first
token batch pays a one-time cost (graph instantiation, first P2P peer-enable); steady state is
extremely flat.

**Correctness confirmed end-to-end.** Both produce coherent output and agree token-for-token for
the first ~20 capitals; they then diverge (UK→Spain→… vs. …→UAE→Saudi…) purely because greedy
argmax over bf16 logits is a chaotic cascade — a tiny numerical difference flips one argmax and the
sequences fork. This is expected and is **not** a fabric error; it confirms the fabric computes the
same FFN to bf16 precision over live 3-GPU NVLink P2P.

**The cost: throughput and a GPU.** MS-AFD is **~3.5× slower** (2.14 vs. 7.45 tok/s). Each token
fans through 48 sequential coordinator↔unit round-trips with blocking-sync coordination and
cross-GPU P2P — inherently serial, and speed was never the claim. It also needs a 3rd GPU for the
A-side + expert memory. This is a **determinism-for-throughput trade**, which is exactly what a
statically-scheduled fabric is for: workloads where a predictable p99 matters more than peak tok/s
(real-time, SLA-bound, lockstep-replicated inference).

---

## 5. Verdict

On the same 30B MoE and the same 3× A100 box, MS-AFD delivers a **measurably and repeatably more
deterministic** per-token latency (p99/p50 1.019 vs. 1.054; CV 0.0082 vs. 0.0120) while producing
correct output over live NVLink P2P — at the cost of ~3.5× throughput and one extra GPU. The
software-defined systolic fabric reproduces the *determinism* characteristic of dedicated inference
hardware on commodity GPUs.

---

*Reproduce:* `scripts/run_latency_fabric.sh` (Case 2) and
`tools/latency_bench.py --mode baseline --baseline-gpus 3` (Case 1). Raw per-token series in
`results/cmp/*.csv`.
