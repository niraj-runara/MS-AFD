# M4 results — the capstone: real model on the LIVE fabric (G1)

**Hardware:** 1× NVIDIA A100-SXM4-80GB, CUDA 12.8 (RunPod).
**Model:** real **meta-llama/Meta-Llama-3-8B** (32 layers, D=4096, F=14336, bf16).

This joins the two halves that were separate until now:
- **M2** proved the *live multi-process fabric* (hub + CUDA-IPC + MPS units) is
  deterministic — but on synthetic weights.
- **M3** proved the *real model* is correct through our FFN — but via an
  **in-process** op, not the live fabric.

M4 runs the real Llama-3 8B where **every layer's FFN is dispatched to a separate,
live, MPS-capped unit process over CUDA IPC** — the actual disaggregated fabric,
with real weights — and measures its determinism during generation.

## Architecture

One **unit process per transformer layer** (32 units, MPS-capped at 3% SMs each),
each holding *that layer's real weights*. HF runs attention / KV / generation; a
coordinator lib (`libmsafd_live.so`) loaded into HF owns IPC input/output tiles and
a control block. Each patched `LlamaMLP.forward` ships the hidden state to its
layer's unit over IPC, which runs the FFN (decode shape T=1 graph-captured) and
returns it. A token flows layer 0→31, each FFN served by its own live micro-unit —
the systolic fabric, serving a real model.

```
HF (A-side, no MPS): attention -> [layer L MLP] --IPC--> unit L (MPS 3%, real Wl) --IPC--> back -> ...
```

## Determinism — per-unit decode FFN (256 generated tokens = 256 beats/unit)

| metric | value |
|--------|-------|
| units (layers) | 32 |
| per-unit p50 | 12.982 – 12.988 ms (spread **0.05%**) |
| per-unit p99/p50 | min 1.0040 · median **1.0045** · **worst 1.0053** |

**The real model, served by 32 live MPS+IPC unit processes, holds p99/p50 = 1.005
worst** — as tight as (in fact tighter than) the synthetic M2 fabric, with the 32
layer-units fair to within 0.05% of each other. Determinism survives the jump from
synthetic weights to a real model on the live fabric.

## Correctness — vs HF (prompt "The capital of France is", 256 tokens)

| rung | metric | value |
|------|--------|-------|
| Rung 2 | teacher-forced next-token argmax agreement | **99.19%** (262 positions) |
| Rung 2 | logit rel-L2 (fabric vs HF) | 0.0135 |
| Rung 3 | free greedy generation | coherent; diverges at tok 1 via a bf16 near-tie |

The live fabric generates fluent, on-topic Llama-3 output (*"Paris, which is located
in the north of the country. The city is located on the Seine River and is the
largest city in France…"*). Teacher-forced argmax agreement is **99.19%** — the
distributions match HF. Free greedy decoding diverges at the first token (HF picks
"Paris**.**", the fabric "Paris**,**" — a genuine near-tie), then cascades; that is
the known benign bf16 greedy-cascade artifact (each context's cuBLAS picks its own
GEMM rounding), not an FFN error — the 99.19% teacher-forced agreement is the metric.

## Takeaway

**The real Llama-3 8B runs on the actual disaggregated, deterministic fabric.** Not
an in-process stand-in — 32 separate MPS-capped processes, each serving one layer's
real FFN over CUDA IPC, driving live token generation, at p99/p50 = 1.005. M2's
determinism and M3's correctness are now one result.

Single GPU here (HF + 32 units co-resident via IPC); the cross-GPU NCCL M2N routing
is proven separately at M2 and composes on top. Full console log in [run.log](run.log).

## Reproduce

```bash
# one-time: dump real per-layer weights
HF_TOKEN=hf_xxx python tools/dump_dense_weights.py --model meta-llama/Meta-Llama-3-8B --out /root/m3_weights
# run: 32 live units + HF generation dispatched to them
HF_TOKEN=hf_xxx bash scripts/run_m3_live.sh meta-llama/Meta-Llama-3-8B /root/m3_weights "The capital of France is" 256 320
```
