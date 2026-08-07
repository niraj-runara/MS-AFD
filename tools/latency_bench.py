#!/usr/bin/env python3
"""Case 1 vs Case 2 latency/determinism benchmark — identical method for both.

Manual KV-cache greedy decode of a real model, timing EACH generated token with
cuda.synchronize() so the two deployments are measured the same way:

  --mode baseline : plain HuggingFace (normal dynamic/pooled inference).
  --mode fabric   : same model, every layer's MoE FFN dispatched to the live
                    MS-AFD fabric (units must already be running; see
                    scripts/run_latency_fabric.sh).

Reports per-token decode latency (p50/p99/p99.9/max/mean±std, p99/p50, tok/s) and
prefill time-to-first-token. Writes the per-token series to --out for plotting.

Usage:
  python tools/latency_bench.py --mode baseline --max-new 256 --out results/cmp/baseline.csv
  (fabric mode is launched by scripts/run_latency_fabric.sh)
"""

import argparse
import ctypes
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def install_fabric(model, args):
    """Patch every MoE block to dispatch its FFN to the live fabric (arity-aware)."""
    lib = ctypes.CDLL(args.lib)
    lib.msafd_live_init.argtypes = [ctypes.c_int] * 6 + [ctypes.c_char_p] * 2
    lib.msafd_live_moe.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.msafd_live_stop.argtypes = []
    cfg = model.config
    D, n_exp, top_k = cfg.hidden_size, cfg.num_experts, cfg.num_experts_per_tok
    norm = bool(getattr(cfg, "norm_topk_prob", True))
    layers = model.model.layers
    n_layers = len(layers)
    dev = f"cuda:{args.device}"   # A-side compute device (experts are off on CPU)

    lib.msafd_live_init(n_layers, D, args.t_max, args.device, n_exp, top_k,
                        args.hdir.encode(), args.ctrl.encode())

    # Probe the block's return arity (bare tensor / (tensor,) / (tensor, router_logits))
    # WITHOUT running the real experts — they're offloaded to CPU, so the true forward
    # would crash on a device mismatch (and we never want HF to compute them anyway).
    # Stub the experts submodule to return an on-device zero tensor of the combined
    # shape [n_tokens, D]; only the block's router/combine plumbing runs.
    from accelerate.hooks import remove_hook_from_module
    dummy = torch.zeros(1, 1, D, dtype=torch.bfloat16, device=dev)
    mlp0 = layers[0].mlp
    n_tok = dummy.shape[0] * dummy.shape[1]
    remove_hook_from_module(mlp0.experts)   # drop the CPU-offload hook; HF never runs these
    orig_fwd = mlp0.experts.forward
    mlp0.experts.forward = lambda *a, **k: torch.zeros(n_tok, D, dtype=dummy.dtype, device=dummy.device)
    try:
        with torch.no_grad():
            probe = type(mlp0).forward(mlp0, dummy)
    finally:
        mlp0.experts.forward = orig_fwd
    probe_len = len(probe) if isinstance(probe, tuple) else 0
    print(f"[fabric] block return arity: probe_len={probe_len}", flush=True)

    def make_patch(L, mlp):
        gate = mlp.gate
        def patched(hidden_states, *a, **kw):
            shp = hidden_states.shape
            h2 = hidden_states.reshape(-1, D).to(torch.bfloat16).contiguous()
            T = h2.shape[0]
            with torch.no_grad():
                lg = gate(h2); lg = lg[0] if isinstance(lg, tuple) else lg
                probs = torch.softmax(lg.float(), dim=-1)
                tw, ti = torch.topk(probs, top_k, dim=-1)
                if norm: tw = tw / tw.sum(dim=-1, keepdim=True)
            ti_c = ti.to(torch.int32).cpu().contiguous()
            tw_c = tw.to(torch.float32).cpu().contiguous()
            out = torch.empty(T, D, dtype=torch.bfloat16, device=h2.device)
            lib.msafd_live_moe(L, ctypes.c_void_p(h2.data_ptr()),
                               ctypes.c_void_p(ti_c.data_ptr()), ctypes.c_void_p(tw_c.data_ptr()),
                               ctypes.c_void_p(out.data_ptr()), T)
            out = out.reshape(shp)
            if probe_len == 0: return out
            if probe_len == 1: return (out,)
            return (out, lg)
        return patched

    for L in range(n_layers):
        layers[L].mlp.forward = make_patch(L, layers[L].mlp)
    return lib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--mode", choices=["baseline", "fabric"], default="baseline")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out", default="results/cmp/latency.csv")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--baseline-gpus", type=int, default=1,
                    help="baseline mode: spread HF across this many GPUs (same N as MS-AFD uses)")
    # fabric-only:
    ap.add_argument("--lib", default="build/libmsafd_moe_live.so")
    ap.add_argument("--hdir", default="/dev/shm/moelive_h")
    ap.add_argument("--ctrl", default="/dev/shm/moelive_ctrl")
    ap.add_argument("--t-max", type=int, default=1024)
    args = ap.parse_args()

    dev = f"cuda:{args.device}"
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.mode == "fabric":
        # MS-AFD: HF never uses its experts (the fabric units do), so offload them
        # to CPU — only the ~3 GB non-expert A-side sits on the GPU. Build the map at
        # LEAF granularity: accelerate stops recursing at the first key it matches, so
        # a broad {"model": gpu} key would swallow the experts before the deeper
        # ".mlp.experts": "cpu" overrides are ever seen (silent OOM). Leaf keys have no
        # descendants, so nothing overrides them — and this also survives the fused
        # Qwen3MoeExperts vs ModuleList drift automatically.
        from transformers import AutoConfig
        from accelerate import init_empty_weights
        cfg = AutoConfig.from_pretrained(args.model)
        with init_empty_weights():
            sk = AutoModelForCausalLM.from_config(cfg)
        device_map = {}
        for name, mod in sk.named_modules():
            if len(list(mod.children())) == 0 and (
                    any(True for _ in mod.parameters(recurse=False))
                    or any(True for _ in mod.buffers(recurse=False))):
                device_map[name] = "cpu" if "mlp.experts" in name else args.device
        del sk
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map=device_map)
    elif args.baseline_gpus > 1:
        # Normal multi-GPU deployment: force HF to shard the model across N GPUs
        # (a token flows through the layers in order — same sequential-across-GPUs
        # shape MS-AFD has). Cap per-GPU memory so it can't all land on one card.
        n = args.baseline_gpus
        cap = max(8, 64 // n + 2)
        mm = {i: f"{cap}GiB" for i in range(n)}
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto", max_memory=mm)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map=dev)
    model.eval()
    lib = install_fabric(model, args) if args.mode == "fabric" else None

    ids = tok(args.prompt, return_tensors="pt").to(dev)
    gen_tokens, times = [], []
    try:
        with torch.no_grad():
            # --- prefill (time-to-first-token) ---
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(**ids, use_cache=True)
            torch.cuda.synchronize()
            ttft = (time.perf_counter() - t0) * 1e3
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            gen_tokens.append(nxt.item())

            # --- decode, timing each token (skip `warmup` from stats) ---
            total = args.max_new - 1
            for i in range(total):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model(nxt, past_key_values=past, use_cache=True)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1e3
                past = out.past_key_values
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                gen_tokens.append(nxt.item())
                if i >= args.warmup:
                    times.append(dt)
    finally:
        if lib is not None:
            lib.msafd_live_stop()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("iter,ms\n")
        for i, m in enumerate(times):
            f.write(f"{i},{m:.6f}\n")

    s = sorted(times)
    p = lambda q: s[int(q * (len(s) - 1))]
    mean = statistics.fmean(times); sd = statistics.pstdev(times)
    print(f"\n===== {args.mode.upper()} | {args.model} | {len(times)} decode tokens =====")
    print(f"generated: {tok.decode(gen_tokens)!r}")
    print(f"TTFT (prefill)      : {ttft:.2f} ms")
    print(f"decode p50 / p99    : {p(0.50):.3f} / {p(0.99):.3f} ms")
    print(f"decode p99.9 / max  : {p(0.999):.3f} / {s[-1]:.3f} ms")
    print(f"decode mean +- sd   : {mean:.3f} +- {sd:.3f} ms  (cv {sd/mean:.4f})")
    print(f"p99/p50 (determinism): {p(0.99)/p(0.50):.4f}")
    print(f"throughput          : {1e3/mean:.2f} tok/s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
