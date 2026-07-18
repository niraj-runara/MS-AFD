#!/usr/bin/env python3
"""M3-mini · reference dumper — the HF oracle.

Runs the real Qwen3-30B-A3B on a prompt, and for ONE MoE layer dumps everything
the C++ fabric needs to reproduce that layer and be checked for correctness:

  meta.json              dims + counts (d_model, d_ff, num_experts, top_k, tokens)
  hidden_in.bin          [T, d_model] bf16   — MoE-block input activations
  topk_idx.bin           [T, top_k] int32    — expert id per token (router top-k)
  topk_w.bin             [T, top_k] float32  — combine weight per (token, expert)
  moe_out.bin            [T, d_model] bf16    — reference MoE-block output
  experts/e{E}_gate.bin  [d_model, d_ff] bf16
  experts/e{E}_up.bin    [d_model, d_ff] bf16
  experts/e{E}_down.bin  [d_ff, d_model] bf16

The C++ side loads the expert weights + hidden_in + routing, runs each expert on
its assigned (capacity-padded) tokens through the fabric, combines per-token with
topk_w, and compares against moe_out.

NOTE: routing is recomputed here exactly as Qwen3 does it (softmax -> top-k ->
optional renorm). It must match the model's config.norm_topk_prob. Verify the
first run's correctness gate before trusting the numbers.

Usage:
  python tools/dump_reference.py --model Qwen/Qwen3-30B-A3B --layer 0 \
      --prompt "The capital of France is" --out results/m3_ref
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def dump_bf16(t: torch.Tensor, path: str) -> None:
    # bf16 is 2 bytes; view as uint16 to get raw bytes numpy can write.
    a = t.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()
    a.tofile(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--layer", type=int, default=0, help="which decoder layer's MoE block")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", default="results/m3_ref")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "experts"), exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    cfg = model.config

    block = model.model.layers[args.layer].mlp   # Qwen3MoeSparseMoeBlock
    d_model = cfg.hidden_size
    d_ff = cfg.moe_intermediate_size
    n_exp = cfg.num_experts
    top_k = cfg.num_experts_per_tok
    norm = bool(getattr(cfg, "norm_topk_prob", True))

    captured = {}

    def hook(_module, inputs, output):
        # input hidden states to the MoE block: [batch, seq, d_model]
        captured["hidden_in"] = inputs[0].detach()
        # Qwen3's MoE block returns (final_hidden_states, router_logits); older/
        # other versions may return just the tensor.
        if isinstance(output, tuple):
            captured["moe_out"] = output[0].detach()
            if len(output) > 1 and torch.is_tensor(output[1]):
                captured["router_logits"] = output[1].detach()
        else:
            captured["moe_out"] = output.detach()

    h = block.register_forward_hook(hook)
    ids = tok(args.prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**ids)
    h.remove()

    hidden = captured["hidden_in"].reshape(-1, d_model)          # [T, d_model]
    moe_out = captured["moe_out"].reshape(-1, d_model)           # [T, d_model]
    T = hidden.shape[0]

    # Router logits: prefer the ones the block actually produced; else recompute.
    with torch.no_grad():
        if "router_logits" in captured:
            logits = captured["router_logits"].reshape(-1, n_exp)
        else:
            g = block.gate(hidden.to(next(block.gate.parameters()).dtype))
            logits = (g[0] if isinstance(g, tuple) else g).reshape(-1, n_exp)
        # Qwen3 routing: softmax over all experts -> top-k -> optional renorm.
        probs = torch.softmax(logits.float(), dim=-1)
        topk_w, topk_idx = torch.topk(probs, top_k, dim=-1)      # [T, top_k]
        if norm:
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    # Dump activations + routing.
    dump_bf16(hidden, os.path.join(args.out, "hidden_in.bin"))
    dump_bf16(moe_out, os.path.join(args.out, "moe_out.bin"))
    topk_idx.to(torch.int32).cpu().numpy().tofile(os.path.join(args.out, "topk_idx.bin"))
    topk_w.to(torch.float32).cpu().numpy().tofile(os.path.join(args.out, "topk_w.bin"))

    # Fused experts (Qwen3MoeExperts): batched weight tensors, [out, in].
    #   gate_up_proj: [E, 2*d_ff, d_model]  (rows 0:d_ff = gate, d_ff:2d_ff = up)
    #   down_proj:    [E, d_model, d_ff]
    # We split per expert and transpose to our row-major [in, out] (matches ffn.cu):
    #   wg,wu = [d_model, d_ff]   wd = [d_ff, d_model]
    # NB: the gate-vs-up row order (gate first) is an assumption — if the C++
    # correctness gate shows a large rel-L2, swap the two halves here.
    gate_up = block.experts.gate_up_proj   # [E, 2*d_ff, d_model]
    down = block.experts.down_proj         # [E, d_model, d_ff]
    assert gate_up.shape == (n_exp, 2 * d_ff, d_model), gate_up.shape
    assert down.shape == (n_exp, d_model, d_ff), down.shape
    for e in range(n_exp):
        gate_w = gate_up[e][:d_ff, :]          # [d_ff, d_model]
        up_w   = gate_up[e][d_ff:2 * d_ff, :]  # [d_ff, d_model]
        down_w = down[e]                        # [d_model, d_ff]
        dump_bf16(gate_w.t(), os.path.join(args.out, f"experts/e{e}_gate.bin"))
        dump_bf16(up_w.t(),   os.path.join(args.out, f"experts/e{e}_up.bin"))
        dump_bf16(down_w.t(), os.path.join(args.out, f"experts/e{e}_down.bin"))

    meta = {
        "model": args.model, "layer": args.layer, "prompt": args.prompt,
        "tokens": int(T), "d_model": int(d_model), "d_intermediate": int(d_ff),
        "num_experts": int(n_exp), "top_k": int(top_k), "norm_topk_prob": norm,
        "dtype": "bf16",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    # Flat meta for the C++ side (no JSON parser needed):
    # tokens d_model d_intermediate num_experts top_k
    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"{T} {d_model} {d_ff} {n_exp} {top_k}\n")

    # Per-expert token load (the dynamic, uneven distribution the fabric must handle).
    counts = torch.bincount(topk_idx.reshape(-1), minlength=n_exp)
    print(f"dumped layer {args.layer}: T={T} tokens, {n_exp} experts, top_k={top_k}")
    print(f"per-expert load: min={counts.min().item()} max={counts.max().item()} "
          f"mean={counts.float().mean().item():.1f}  -> capacity C must be >= max")
    print(f"wrote {args.out}/  (meta.json, hidden_in, topk_*, moe_out, experts/)")


if __name__ == "__main__":
    main()
