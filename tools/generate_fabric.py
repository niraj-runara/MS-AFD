#!/usr/bin/env python3
"""M3-full Rungs 2-3 — real generation with the MoE FFN served by our fabric op.

HF runs the real Qwen3-30B-A3B (attention, KV cache, router, sampling); we
monkeypatch every MoE block so its expert FFN is computed by our shared-library
op (build/libmsafd_moe.so — the fabric's own GEMM+SwiGLU math). Then we generate
greedily and compare the token sequence to pure-HF generation.

  Rung 2 = the first forward is correct (per-layer FFN via our op).
  Rung 3 = the full generated sequence matches HF token-for-token.

Usage:
  python tools/generate_fabric.py --model Qwen/Qwen3-30B-A3B \
      --prompt "The capital of France is" --max-new 20 \
      --lib build/libmsafd_moe.so
"""

import argparse
import ctypes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--lib", default="build/libmsafd_moe.so")
    args = ap.parse_args()

    lib = ctypes.CDLL(args.lib)
    lib.msafd_moe_ffn.restype = None
    lib.msafd_moe_ffn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    cfg = model.config
    D, F = cfg.hidden_size, cfg.moe_intermediate_size
    E, top_k = cfg.num_experts, cfg.num_experts_per_tok
    norm = bool(getattr(cfg, "norm_topk_prob", True))
    layers = model.model.layers
    ids = tok(args.prompt, return_tensors="pt").to("cuda")

    # --- reference: pure HF greedy generation --------------------------------
    with torch.no_grad():
        ref = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False)
    ref_ids = ref[0].tolist()
    print("reference (HF):", tok.decode(ref[0][ids.input_ids.shape[1]:]))

    # --- probe the MoE block's return type, then patch every layer -----------
    dummy = torch.zeros(1, 1, D, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        probe = type(layers[0].mlp).forward(layers[0].mlp, dummy)
    returns_tuple = isinstance(probe, tuple)

    def make_patch(mlp):
        gate, gate_up, down = mlp.gate, mlp.experts.gate_up_proj, mlp.experts.down_proj

        def patched(hidden_states, *a, **kw):
            shp = hidden_states.shape
            h2 = hidden_states.reshape(-1, D).contiguous()
            T = h2.shape[0]
            with torch.no_grad():
                lg = gate(h2)
                lg = lg[0] if isinstance(lg, tuple) else lg
                probs = torch.softmax(lg.float(), dim=-1)
                tw, ti = torch.topk(probs, top_k, dim=-1)
                if norm:
                    tw = tw / tw.sum(dim=-1, keepdim=True)
            ti_c = ti.to(torch.int32).cpu().contiguous()
            tw_c = tw.to(torch.float32).cpu().contiguous()
            h2bf = h2.to(torch.bfloat16).contiguous()
            out = torch.empty(T, D, dtype=torch.bfloat16, device=h2.device)
            lib.msafd_moe_ffn(
                ctypes.c_void_p(h2bf.data_ptr()),
                ctypes.c_void_p(gate_up.data_ptr()),
                ctypes.c_void_p(down.data_ptr()),
                ctypes.c_void_p(ti_c.data_ptr()),
                ctypes.c_void_p(tw_c.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                T, D, F, E, top_k,
            )
            out = out.reshape(shp)
            return (out, lg) if returns_tuple else out

        return patched

    for L in range(len(layers)):
        layers[L].mlp.forward = make_patch(layers[L].mlp)

    # --- fabric-FFN greedy generation ----------------------------------------
    with torch.no_grad():
        fab = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False)
    fab_ids = fab[0].tolist()
    print("fabric-FFN   :", tok.decode(fab[0][ids.input_ids.shape[1]:]))

    # --- compare -------------------------------------------------------------
    n = min(len(ref_ids), len(fab_ids))
    match = sum(1 for i in range(n) if ref_ids[i] == fab_ids[i])
    gen0 = ids.input_ids.shape[1]
    new_ref, new_fab = ref_ids[gen0:], fab_ids[gen0:]
    new_match = sum(1 for i in range(min(len(new_ref), len(new_fab)))
                    if new_ref[i] == new_fab[i])
    print(f"\ntokens matched (full seq): {match}/{n}")
    print(f"generated tokens matched : {new_match}/{len(new_ref)}")
    if new_ref == new_fab:
        print("PASS — fabric-FFN generation is token-identical to HF")
    else:
        first = next(i for i in range(len(new_ref))
                     if i >= len(new_fab) or new_ref[i] != new_fab[i])
        print(f"first divergence at generated token {first}: "
              f"HF={new_ref[first]} fabric={new_fab[first] if first < len(new_fab) else 'NA'}")
        print("(small divergence late in the sequence can be bf16 rounding; "
              "early divergence means the FFN/router path is off)")


if __name__ == "__main__":
    main()
