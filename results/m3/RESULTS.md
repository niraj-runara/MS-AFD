# M3 results — the real model, end to end (dense)

**Hardware:** 1× NVIDIA A100-SXM4-80GB, CUDA 12.8 (RunPod).
**Model:** real **meta-llama/Meta-Llama-3-8B** (32 layers, d_model=4096,
d_intermediate=14336, bf16), downloaded from Hugging Face.

HuggingFace runs everything a fabric doesn't test — attention, the KV cache, and
generation. **Every decoder layer's FFN is computed by our op**
(`build/libmsafd_dense.so`, the same GEMM + SwiGLU as the M2 micro-unit), loaded
with the layer's real weights and called from HF via ctypes.

## Why dense M3 is simpler than MoE M3

The MoE track had to solve a *data-dependent routing* problem (per-expert token
counts vary each pass) with fixed-capacity zero-padding to keep tiles fixed-shape.
A **dense** FFN is already fixed-shape — every token goes through the one FFN — so
there is no router, no capacity padding, no scatter-combine. The op is just the
SwiGLU FFN over all tokens: `out = down( silu(gate(x)) * up(x) )`.

## Rung 1 — all-layers correctness

One HF forward captures each layer's MLP input and output; the fabric op recomputes
the MLP from the real weights and we measure relative-L2 vs HF, per layer.

| metric | value |
|--------|-------|
| layers checked | **32 / 32** |
| rel-L2 min / mean / max | 0.00003 / **0.00316** / 0.00383 |
| worst layer | L15 @ 0.00383 |
| result | **PASS** (tol 0.05) |

The error is **flat with depth** (~0.003 at every layer, no upward trend) — the
fabric FFN reproduces HF for the whole model and error does not compound.

## Rungs 2-3 — generation vs HF

Prompt: *"The capital of France is"*, 20 new tokens, greedy.

| rung | metric | value |
|------|--------|-------|
| Rung 2 | teacher-forced next-token argmax agreement | **96.15%** |
| Rung 2 | logit rel-L2 (fabric vs HF, teacher-forced) | 0.01035 |
| Rung 3 | greedy generated tokens matching HF | **20 / 20** |

- **Rung 3 is token-identical to HF** — with every FFN served by the fabric op, the
  model generates exactly *"Paris. It is located in the north of the country. The
  city is situated on the banks of"*.
- **Rung 2** confirms the distributions agree: logit rel-L2 = 0.010 (the per-layer
  ~0.003 compounded over 32 layers), far too small to move a non-tie argmax. The
  single teacher-forced disagreement (96.15% = 25/26 positions) is a benign bf16
  near-tie — it does not appear on the greedy path, which is why Rung 3 is exact.

## Takeaway

**The real Llama-3 8B generates token-correct output with every FFN executed by the
fabric's own implementation.** Attention on the HF A-side (the plan permits a real
attention impl there); the FFN — the part the systolic-fabric thesis is about —
served by the fabric op, correct against HF across all 32 layers and token-for-token
through generation. The LPU-style *determinism* is proven at M1/M2; M3's role is
end-to-end correctness, met here.

Full console log in [run.log](run.log).

## Reproduce

```bash
# needs a HF token with access to the gated Llama-3 weights
HF_TOKEN=hf_xxx bash scripts/run_m3_dense.sh meta-llama/Meta-Llama-3-8B "The capital of France is" 20
# or the two rungs directly:
python tools/check_all_layers_dense.py --model meta-llama/Meta-Llama-3-8B --lib build/libmsafd_dense.so
python tools/generate_dense.py       --model meta-llama/Meta-Llama-3-8B --lib build/libmsafd_dense.so --max-new 20
```
