#!/usr/bin/env python3
"""M3 (dense) · Rung 1 — all-layers correctness.

One HF forward captures every decoder layer's MLP input and output; then, layer by
layer, the fabric's dense-FFN op (build/libmsafd_dense.so) recomputes that layer's
MLP from the real weights and we measure relative-L2 vs HF. Proves the fabric
serves the WHOLE real model correctly and that error doesn't compound with depth.

Usage:
  python tools/check_all_layers_dense.py --model meta-llama/Meta-Llama-3-8B \
      --prompt "The capital of France is" --lib build/libmsafd_dense.so
"""

import argparse
import ctypes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--lib", default="build/libmsafd_dense.so")
    ap.add_argument("--tol", type=float, default=0.05)
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
    n_layers = len(layers)
    print(f"model {args.model}: {n_layers} layers, D={D} F={F}")

    # One forward, capture each layer's MLP input[0] and output.
    cap = {}

    def make_hook(L):
        def hook(_m, inp, out):
            cap[L] = (inp[0].detach().reshape(-1, D), out.detach().reshape(-1, D))
        return hook

    handles = [layers[L].mlp.register_forward_hook(make_hook(L)) for L in range(n_layers)]
    ids = tok(args.prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**ids)
    for h in handles:
        h.remove()

    results = []
    for L in range(n_layers):
        hidden, ref = cap[L]
        T = hidden.shape[0]
        mlp = layers[L].mlp
        gw, uw, dw = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight

        h_in = hidden.to(torch.bfloat16).contiguous()
        out = torch.empty(T, D, dtype=torch.bfloat16, device="cuda")
        lib.msafd_dense_ffn(
            ctypes.c_void_p(h_in.data_ptr()),
            ctypes.c_void_p(gw.contiguous().data_ptr()),
            ctypes.c_void_p(uw.contiguous().data_ptr()),
            ctypes.c_void_p(dw.contiguous().data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            T, D, F,
        )
        rel = ((out.float() - ref.float()).norm() / (ref.float().norm() + 1e-9)).item()
        results.append(rel)
        print(f"layer {L:2d}: rel-L2 vs HF = {rel:.5f}")

    print("\n=== all-layers correctness ===")
    print(f"layers checked      : {len(results)}")
    print(f"rel-L2 min/mean/max : {min(results):.5f} / "
          f"{sum(results)/len(results):.5f} / {max(results):.5f}")
    worst = max(range(len(results)), key=lambda i: results[i])
    print(f"worst layer         : L{worst} @ {results[worst]:.5f}")
    print("PASS" if max(results) <= args.tol else f"FAIL (some layer exceeds {args.tol})")


if __name__ == "__main__":
    main()
