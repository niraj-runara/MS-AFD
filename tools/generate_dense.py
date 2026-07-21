#!/usr/bin/env python3
"""M3 (dense) · Rungs 2-3 — real generation with the FFN served by the fabric op.

HF runs the real Llama-3 8B (attention, KV cache, sampling); we monkeypatch every
decoder layer's MLP so its SwiGLU FFN is computed by our shared-library op
(build/libmsafd_dense.so — the fabric's own GEMM+SwiGLU math). Then:

  Rung 2 = teacher-forced next-token argmax agreement with pure HF (the real
           correctness metric; factors out the greedy-decoding divergence cascade).
  Rung 3 = free greedy generation compared to HF token-for-token.

Usage:
  python tools/generate_dense.py --model meta-llama/Meta-Llama-3-8B \
      --prompt "The capital of France is" --max-new 20 --lib build/libmsafd_dense.so
"""

import argparse
import ctypes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--lib", default="build/libmsafd_dense.so")
    args = ap.parse_args()

    lib = ctypes.CDLL(args.lib)
    lib.msafd_dense_ffn.restype = None
    lib.msafd_dense_ffn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    cfg = model.config
    D, F = cfg.hidden_size, cfg.intermediate_size
    layers = model.model.layers
    ids = tok(args.prompt, return_tensors="pt").to("cuda")

    # --- reference: pure HF greedy generation + teacher-forced logits --------
    with torch.no_grad():
        ref = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False)
        hf_logits = model(ref).logits[0].float().cpu()   # [S, vocab] — pre-patch
    ref_ids = ref[0].tolist()
    gen0 = ids.input_ids.shape[1]
    print("reference (HF):", tok.decode(ref[0][gen0:]))

    # --- probe MLP return type, then patch every layer ------------------------
    dummy = torch.zeros(1, 1, D, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        probe = type(layers[0].mlp).forward(layers[0].mlp, dummy)
    returns_tuple = isinstance(probe, tuple)

    def make_patch(mlp):
        gw, uw, dw = (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)

        def patched(hidden_states, *a, **kw):
            shp = hidden_states.shape
            h2 = hidden_states.reshape(-1, D).to(torch.bfloat16).contiguous()
            T = h2.shape[0]
            out = torch.empty(T, D, dtype=torch.bfloat16, device=h2.device)
            lib.msafd_dense_ffn(
                ctypes.c_void_p(h2.data_ptr()),
                ctypes.c_void_p(gw.contiguous().data_ptr()),
                ctypes.c_void_p(uw.contiguous().data_ptr()),
                ctypes.c_void_p(dw.contiguous().data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                T, D, F,
            )
            out = out.reshape(shp)
            return (out,) if returns_tuple else out

        return patched

    for L in range(len(layers)):
        layers[L].mlp.forward = make_patch(layers[L].mlp)

    # --- Rung 2: teacher-forced correctness (same tokens, compare distributions)
    with torch.no_grad():
        fab_logits = model(ref).logits[0].float().cpu()
    hf_arg, fab_arg = hf_logits.argmax(-1), fab_logits.argmax(-1)
    agree = (hf_arg == fab_arg).float().mean().item()
    rel = ((hf_logits - fab_logits).norm() / (hf_logits.norm() + 1e-9)).item()
    print(f"\n[Rung 2] teacher-forced next-token argmax agreement "
          f"{agree * 100:.2f}%  |  logit rel-L2 {rel:.5f}")

    # --- Rung 3: fabric-FFN greedy generation vs HF ---------------------------
    with torch.no_grad():
        fab = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False)
    fab_ids = fab[0].tolist()
    print("fabric-FFN   :", tok.decode(fab[0][gen0:]))

    new_ref, new_fab = ref_ids[gen0:], fab_ids[gen0:]
    new_match = sum(1 for i in range(min(len(new_ref), len(new_fab)))
                    if new_ref[i] == new_fab[i])
    print(f"\n[Rung 3] generated tokens matched: {new_match}/{len(new_ref)}")
    if new_ref == new_fab:
        print("PASS — fabric-FFN generation is token-identical to HF")
    else:
        first = next(i for i in range(len(new_ref))
                     if i >= len(new_fab) or new_ref[i] != new_fab[i])
        print(f"first divergence at generated token {first}: "
              f"HF={new_ref[first]} fabric={new_fab[first] if first < len(new_fab) else 'NA'}")
        print("(late divergence is typically bf16 rounding in a greedy near-tie; "
              "Rung 2's teacher-forced agreement is the real correctness metric)")


if __name__ == "__main__":
    main()
