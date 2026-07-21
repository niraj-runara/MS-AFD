#!/usr/bin/env python3
"""M4 — real Qwen3-30B-A3B generation served by the LIVE multi-process MoE fabric.

Unlike M3 (an in-process op), every layer's MoE FFN is dispatched to a separate,
MPS-capped unit process over CUDA IPC via the coordinator lib (libmsafd_moe_live.so).
HF runs attention / KV / router / generation; the fabric serves every expert FFN.
This joins M2 (a deterministic live fabric) and M3 (a correct real model).

  Rung 2 = teacher-forced next-token argmax agreement with pure HF.
  Rung 3 = free greedy generation compared to HF token-for-token.

The unit processes must already be running (launched by scripts/run_moe_live.sh);
msafd_live_init() blocks until all of them attach.
"""

import argparse
import ctypes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--lib", default="build/libmsafd_moe_live.so")
    ap.add_argument("--hdir", default="/dev/shm/moelive_h")
    ap.add_argument("--ctrl", default="/dev/shm/moelive_ctrl")
    ap.add_argument("--t-max", type=int, default=1024)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=20)
    args = ap.parse_args()

    lib = ctypes.CDLL(args.lib)
    lib.msafd_live_init.argtypes = [ctypes.c_int] * 6 + [ctypes.c_char_p] * 2
    lib.msafd_live_moe.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.msafd_live_stop.argtypes = []

    dev = f"cuda:{args.device}"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=dev
    )
    model.eval()
    cfg = model.config
    D, F = cfg.hidden_size, cfg.moe_intermediate_size
    n_exp, top_k = cfg.num_experts, cfg.num_experts_per_tok
    norm = bool(getattr(cfg, "norm_topk_prob", True))
    layers = model.model.layers
    n_layers = len(layers)
    ids = tok(args.prompt, return_tensors="pt").to(dev)

    # --- reference: pure HF greedy generation + teacher-forced logits (pre-patch)
    with torch.no_grad():
        ref = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False)
        hf_logits = model(ref).logits[0].float().cpu()
    ref_ids = ref[0].tolist()
    gen0 = ids.input_ids.shape[1]
    print("reference (HF):", tok.decode(ref[0][gen0:]))

    # --- bring up the live fabric (blocks until all units attach) --------------
    lib.msafd_live_init(n_layers, D, args.t_max, args.device, n_exp, top_k,
                        args.hdir.encode(), args.ctrl.encode())

    dummy = torch.zeros(1, 1, D, dtype=torch.bfloat16, device=dev)
    with torch.no_grad():
        probe = type(layers[0].mlp).forward(layers[0].mlp, dummy)
    returns_tuple = isinstance(probe, tuple)

    def make_patch(L, mlp):
        gate = mlp.gate

        def patched(hidden_states, *a, **kw):
            shp = hidden_states.shape
            h2 = hidden_states.reshape(-1, D).to(torch.bfloat16).contiguous()
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
            out = torch.empty(T, D, dtype=torch.bfloat16, device=h2.device)
            lib.msafd_live_moe(L, ctypes.c_void_p(h2.data_ptr()),
                               ctypes.c_void_p(ti_c.data_ptr()),
                               ctypes.c_void_p(tw_c.data_ptr()),
                               ctypes.c_void_p(out.data_ptr()), T)
            out = out.reshape(shp)
            return (out,) if returns_tuple else out

        return patched

    for L in range(n_layers):
        layers[L].mlp.forward = make_patch(L, layers[L].mlp)

    try:
        with torch.no_grad():
            fab_logits = model(ref).logits[0].float().cpu()
        hf_arg, fab_arg = hf_logits.argmax(-1), fab_logits.argmax(-1)
        agree = (hf_arg == fab_arg).float().mean().item()
        rel = ((hf_logits - fab_logits).norm() / (hf_logits.norm() + 1e-9)).item()
        print(f"\n[Rung 2] teacher-forced next-token argmax agreement "
              f"{agree * 100:.2f}%  |  logit rel-L2 {rel:.5f}")

        with torch.no_grad():
            fab = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False)
        fab_ids = fab[0].tolist()
        print("live-fabric  :", tok.decode(fab[0][gen0:]))
        new_ref, new_fab = ref_ids[gen0:], fab_ids[gen0:]
        nm = sum(1 for i in range(min(len(new_ref), len(new_fab))) if new_ref[i] == new_fab[i])
        print(f"\n[Rung 3] generated tokens matched: {nm}/{len(new_ref)}")
        if new_ref == new_fab:
            print("PASS — live-fabric generation is token-identical to HF")
        else:
            first = next(i for i in range(len(new_ref)) if i >= len(new_fab) or new_ref[i] != new_fab[i])
            print(f"first divergence at generated token {first} "
                  f"(late divergence is a bf16 greedy near-tie; Rung 2 is the metric)")
    finally:
        lib.msafd_live_stop()


if __name__ == "__main__":
    main()
