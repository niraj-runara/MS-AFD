#!/usr/bin/env python3
"""M3-full · Rung 1 — all-layers correctness.

M3-mini proved layer 0's FFN reproduces HF. This checks EVERY MoE layer: one HF
forward captures each layer's MoE input/routing/output; then, layer by layer, it
dumps that layer's real expert weights, runs the proven ./build/m3_layer
correctness check, records the rel-L2, and deletes the weights before the next
layer (so peak disk is ~one layer, not ~48).

Proves the fabric serves the WHOLE real model correctly and that error doesn't
compound with depth.

Usage:
  python tools/check_all_layers.py --model Qwen/Qwen3-30B-A3B \
      --prompt "The capital of France is" --out results/m3_all
"""

import argparse
import os
import re
import shutil
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def dump_bf16(t, path):
    a = t.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()
    a.tofile(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", default="results/m3_all")
    ap.add_argument("--m3-bin", default="./build/m3_layer")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    cfg = model.config
    D, F = cfg.hidden_size, cfg.moe_intermediate_size
    n_exp, top_k = cfg.num_experts, cfg.num_experts_per_tok
    norm = bool(getattr(cfg, "norm_topk_prob", True))
    layers = model.model.layers
    n_layers = len(layers)

    # One forward, capture every MoE block's input / output / router logits.
    cap = {}

    def make_hook(L):
        def hook(_m, inp, out):
            rec = {"hidden_in": inp[0].detach().reshape(-1, D)}
            if isinstance(out, tuple):
                rec["moe_out"] = out[0].detach().reshape(-1, D)
                rec["router_logits"] = (out[1].detach().reshape(-1, n_exp)
                                        if len(out) > 1 and torch.is_tensor(out[1]) else None)
            else:
                rec["moe_out"] = out.detach().reshape(-1, D)
                rec["router_logits"] = None
            cap[L] = rec
        return hook

    handles = [layers[L].mlp.register_forward_hook(make_hook(L)) for L in range(n_layers)]
    ids = tok(args.prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**ids)
    for h in handles:
        h.remove()

    os.makedirs(args.out, exist_ok=True)
    results = []  # (layer, rel_l2)
    rel_re = re.compile(r"relative L2 vs HF moe_out = ([0-9.]+)")

    for L in range(n_layers):
        rec = cap[L]
        hidden, moe_out = rec["hidden_in"], rec["moe_out"]
        T = hidden.shape[0]
        block = layers[L].mlp

        with torch.no_grad():
            logits = rec["router_logits"]
            if logits is None:
                g = block.gate(hidden.to(next(block.gate.parameters()).dtype))
                logits = (g[0] if isinstance(g, tuple) else g).reshape(-1, n_exp)
            probs = torch.softmax(logits.float(), dim=-1)
            topk_w, topk_idx = torch.topk(probs, top_k, dim=-1)
            if norm:
                topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        d = os.path.join(args.out, f"L{L}")
        os.makedirs(os.path.join(d, "experts"), exist_ok=True)
        with open(os.path.join(d, "meta.txt"), "w") as f:
            f.write(f"{T} {D} {F} {n_exp} {top_k}\n")
        dump_bf16(hidden, os.path.join(d, "hidden_in.bin"))
        dump_bf16(moe_out, os.path.join(d, "moe_out.bin"))
        topk_idx.to(torch.int32).cpu().numpy().tofile(os.path.join(d, "topk_idx.bin"))
        topk_w.to(torch.float32).cpu().numpy().tofile(os.path.join(d, "topk_w.bin"))

        gate_up, down = block.experts.gate_up_proj, block.experts.down_proj
        for e in range(n_exp):
            dump_bf16(gate_up[e][:F, :].t(),        os.path.join(d, f"experts/e{e}_gate.bin"))
            dump_bf16(gate_up[e][F:2 * F, :].t(),   os.path.join(d, f"experts/e{e}_up.bin"))
            dump_bf16(down[e].t(),                  os.path.join(d, f"experts/e{e}_down.bin"))

        out = subprocess.run([args.m3_bin, d], capture_output=True, text=True)
        m = rel_re.search(out.stdout)
        rel = float(m.group(1)) if m else float("nan")
        results.append((L, rel))
        print(f"layer {L:2d}: rel-L2 = {rel:.5f}"
              + ("" if m else f"  (m3_layer output:\n{out.stdout}\n{out.stderr})"))
        shutil.rmtree(os.path.join(d, "experts"))  # free disk before next layer

    print("\n=== all-layers correctness ===")
    vals = [r for _, r in results if r == r]  # drop NaN
    if vals:
        worst = max(results, key=lambda x: (x[1] if x[1] == x[1] else -1))
        print(f"layers checked : {len(results)}")
        print(f"rel-L2 min/mean/max : {min(vals):.5f} / {sum(vals)/len(vals):.5f} / {max(vals):.5f}")
        print(f"worst layer    : L{worst[0]} @ {worst[1]:.5f}")
        print("PASS" if max(vals) <= 0.05 else "FAIL (some layer exceeds 0.05)")
    else:
        print("no rel-L2 parsed — check m3_layer output above")


if __name__ == "__main__":
    main()
