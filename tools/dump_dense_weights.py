#!/usr/bin/env python3
"""M4 (G1) — dump each dense layer's real FFN weights for the live units.

Loads the HF model and writes, per layer L, three raw bf16 files in HF nn.Linear
layout: L{L}_gate.bin [F,D], L{L}_up.bin [F,D], L{L}_down.bin [D,F]. Also writes
config.txt ("n_layers D F") so the launcher can size the unit processes.

Usage:
  python tools/dump_dense_weights.py --model meta-llama/Meta-Llama-3-8B --out /root/m3_weights
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
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--out", default="/root/m3_weights")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu"
    )
    cfg = model.config
    D, F, n = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.txt"), "w") as f:
        f.write(f"{n} {D} {F}\n")

    layers = model.model.layers
    for L in range(n):
        mlp = layers[L].mlp
        base = os.path.join(args.out, f"L{L}")
        dump_bf16(mlp.gate_proj.weight, base + "_gate.bin")   # [F, D]
        dump_bf16(mlp.up_proj.weight,   base + "_up.bin")     # [F, D]
        dump_bf16(mlp.down_proj.weight, base + "_down.bin")   # [D, F]
        print(f"dumped layer {L:2d}/{n}", flush=True)

    print(f"done: {n} layers -> {args.out}  (D={D} F={F})")


if __name__ == "__main__":
    main()
