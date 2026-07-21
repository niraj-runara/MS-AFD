#!/usr/bin/env python3
"""M4 — dump every layer's real MoE expert weights for the live units.

Writes, per layer L, the fused expert weights in HF layout (what moe_live_unit
loads and moe_ffn_op's gemm_linear consumes directly):
  <out>/L{L}_gate_up.bin   [n_exp, 2*F, D] bf16
  <out>/L{L}_down.bin       [n_exp, D, F]   bf16
plus <out>/config.txt : "n_layers D F n_exp top_k"

NB: this is large (~58 GB for Qwen3-30B-A3B). The box needs enough disk for the
model cache (~61 GB) AND this dump.

Usage: python tools/dump_moe_weights.py --model Qwen/Qwen3-30B-A3B --out /workspace/m4_weights
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM


def dump_bf16(t, path):
    a = t.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()
    a.tofile(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--out", default="/workspace/m4_weights")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    cfg = model.config
    D, F = cfg.hidden_size, cfg.moe_intermediate_size
    n_exp, top_k = cfg.num_experts, cfg.num_experts_per_tok
    layers = model.model.layers
    n_layers = len(layers)

    for L in range(n_layers):
        ex = layers[L].mlp.experts
        dump_bf16(ex.gate_up_proj, os.path.join(args.out, f"L{L}_gate_up.bin"))
        dump_bf16(ex.down_proj,    os.path.join(args.out, f"L{L}_down.bin"))
        print(f"dumped layer {L}/{n_layers-1}", flush=True)

    with open(os.path.join(args.out, "config.txt"), "w") as f:
        f.write(f"{n_layers} {D} {F} {n_exp} {top_k}\n")
    print(f"done: {n_layers} layers, D={D} F={F} n_exp={n_exp} top_k={top_k} -> {args.out}")


if __name__ == "__main__":
    main()
