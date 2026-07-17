#!/usr/bin/env python3
"""M1 — plot determinism vs slice count from results/m1/summary.csv.

Two panels, single y-axis each (no dual-axis):
  left  — p99/p50 (worst slice) vs N: the determinism metric
  right — p50 (median slice) vs N: absolute per-iter cost (context)

Two categorical series: partitioned (100/N% each) vs oversubscribed (fixed 10%).
Colors are the validated blue/orange pair (CVD-safe); identity also carried by the
legend and direct end-labels. Renders light + dark PNGs.

Usage: plot_m1.py [summary.csv] [out_dir]
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- design tokens (from the dataviz reference palette) ----------------------
THEMES = {
    "light": dict(
        surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
        muted="#898781", grid="#e1e0d9", baseline="#c3c2b7",
        s_part="#2a78d6", s_over="#eb6834",
    ),
    "dark": dict(
        surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
        muted="#898781", grid="#2c2c2a", baseline="#383835",
        s_part="#3987e5", s_over="#d95926",
    ),
}


def load(path):
    part, over = {}, {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            n = int(r["slices"])
            rec = (n, float(r["ratio_worst"]), float(r["slice_p50_ms_median"]))
            (part if r["mode"] == "partitioned" else over)[n] = rec
    p = [part[k] for k in sorted(part)]
    o = [over[k] for k in sorted(over)]
    return p, o


def style_ax(ax, t, xs):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["baseline"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.grid(True, color=t["grid"], linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    ax.set_xticks(xs)
    ax.set_xlabel("slices on one GPU (N)", color=t["secondary"], fontsize=10)


def plot(path, out_dir, mode):
    t = THEMES[mode]
    part, over = load(path)
    xs = [r[0] for r in part]

    plt.rcParams["font.family"] = ["DejaVu Sans"]  # available; matches system sans
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.0), dpi=160)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.22,
                        wspace=0.24)
    fig.patch.set_facecolor(t["surface"])

    lp = dict(linewidth=2.0, marker="o", markersize=7, markeredgewidth=1.5,
              markeredgecolor=t["surface"], clip_on=False, zorder=3)

    # --- left: p99/p50 vs N ---
    style_ax(axL, t, xs)
    axL.axhline(1.0, color=t["muted"], linewidth=1.0, linestyle=(0, (4, 4)),
                alpha=0.7, zorder=1)
    axL.text(xs[0], 1.0, "perfect determinism (1.0)", color=t["muted"],
             fontsize=8, va="bottom", ha="left")
    axL.plot([r[0] for r in part], [r[1] for r in part], color=t["s_part"], **lp)
    axL.plot([r[0] for r in over], [r[1] for r in over], color=t["s_over"], **lp)
    axL.set_ylabel("per-slice  p99 / p50  (worst)", color=t["secondary"], fontsize=10)
    axL.set_title("Determinism holds as the GPU is sliced",
                  color=t["primary"], fontsize=12, fontweight="bold", loc="left")
    # direct series labels + selective value labels at the N=48 endpoints
    axL.annotate(f"partitioned  {part[-1][1]:.3f}", (part[-1][0], part[-1][1]),
                 textcoords="offset points", xytext=(-8, 8),
                 color=t["secondary"], fontsize=9, ha="right")
    axL.annotate(f"oversubscribed  {over[-1][1]:.3f}", (over[-1][0], over[-1][1]),
                 textcoords="offset points", xytext=(-8, -6),
                 color=t["secondary"], fontsize=9, ha="right", va="top")

    # --- right: p50 vs N ---
    style_ax(axR, t, xs)
    axR.plot([r[0] for r in part], [r[2] for r in part], color=t["s_part"], **lp)
    axR.plot([r[0] for r in over], [r[2] for r in over], color=t["s_over"], **lp)
    axR.set_ylabel("per-slice  p50  latency (ms)", color=t["secondary"], fontsize=10)
    axR.set_title("Cost per slice (we optimize stability, not speed)",
                  color=t["primary"], fontsize=12, fontweight="bold", loc="left")

    # --- legend (identity: colored swatch + ink label) ---
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=t["s_part"], lw=2.5, marker="o", markersize=7,
               markeredgecolor=t["surface"], label="partitioned  (100/N % SMs, Σ≤100%)"),
        Line2D([0], [0], color=t["s_over"], lw=2.5, marker="o", markersize=7,
               markeredgecolor=t["surface"], label="oversubscribed  (fixed 10% each)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=9, labelcolor=t["secondary"])

    fig.suptitle("MS-AFD M1 — per-slice determinism vs slice count  (1× A100 80GB, Llama-3 8B FFN)",
                 color=t["primary"], fontsize=13, fontweight="bold", y=0.95)

    out = f"{out_dir}/m1_determinism{'' if mode == 'light' else '_dark'}.png"
    fig.savefig(out, facecolor=t["surface"], dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/m1/summary.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results/m1"
    for mode in ("light", "dark"):
        plot(path, out_dir, mode)


if __name__ == "__main__":
    main()
