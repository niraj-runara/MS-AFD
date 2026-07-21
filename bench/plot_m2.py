#!/usr/bin/env python3
"""M2 (dense) — plot fabric determinism vs micro-unit count.

Two panels, single y-axis each:
  left  — p99/p50 vs N: determinism, two series
            * per-unit FFN compute (the M1-comparable metric)
            * whole-fabric A-side round-trip beat (NCCL scatter -> IPC fan-out ->
              gather), which carries the cross-GPU hop
  right — p50 vs N: absolute per-beat cost (context; we optimize stability)

Colors are the validated blue/orange pair (CVD-safe); identity also carried by the
legend and direct end-labels. Renders light + dark PNGs.

Usage: plot_m2.py [summary.csv] [summary_aside.csv] [out_dir]
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": dict(
        surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
        muted="#898781", grid="#e1e0d9", baseline="#c3c2b7",
        s_unit="#2a78d6", s_beat="#eb6834",
    ),
    "dark": dict(
        surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
        muted="#898781", grid="#2c2c2a", baseline="#383835",
        s_unit="#3987e5", s_beat="#d95926",
    ),
}


def load(unit_path, beat_path):
    unit, beat = {}, {}
    with open(unit_path, newline="") as fh:
        for r in csv.DictReader(fh):
            n = int(r["slices"])
            unit[n] = (n, float(r["ratio_worst"]), float(r["slice_p50_ms_median"]))
    with open(beat_path, newline="") as fh:
        for r in csv.DictReader(fh):
            n = int(r["slices"])
            beat[n] = (n, float(r["rt_ratio"]), float(r["rt_p50_ms"]))
    u = [unit[k] for k in sorted(unit)]
    b = [beat[k] for k in sorted(beat)]
    return u, b


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
    ax.set_xlabel("micro-units per F-side GPU (N)", color=t["secondary"], fontsize=10)


def plot(unit_path, beat_path, out_dir, mode):
    t = THEMES[mode]
    unit, beat = load(unit_path, beat_path)
    xs = [r[0] for r in unit]

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.0), dpi=160)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.22, wspace=0.24)
    fig.patch.set_facecolor(t["surface"])

    lp = dict(linewidth=2.0, marker="o", markersize=7, markeredgewidth=1.5,
              markeredgecolor=t["surface"], clip_on=False, zorder=3)

    # --- left: p99/p50 vs N ---
    style_ax(axL, t, xs)
    axL.axhline(1.0, color=t["muted"], linewidth=1.0, linestyle=(0, (4, 4)),
                alpha=0.7, zorder=1)
    axL.text(xs[0], 1.0, "perfect determinism (1.0)", color=t["muted"],
             fontsize=8, va="bottom", ha="left")
    axL.plot([r[0] for r in unit], [r[1] for r in unit], color=t["s_unit"], **lp)
    axL.plot([r[0] for r in beat], [r[1] for r in beat], color=t["s_beat"], **lp)
    axL.set_ylabel("p99 / p50  (worst)", color=t["secondary"], fontsize=10)
    axL.set_title("Determinism holds across the disaggregated fabric",
                  color=t["primary"], fontsize=12, fontweight="bold", loc="left")
    axL.annotate(f"per-unit FFN  {unit[-1][1]:.3f}", (unit[-1][0], unit[-1][1]),
                 textcoords="offset points", xytext=(-8, -6),
                 color=t["secondary"], fontsize=9, ha="right", va="top")
    axL.annotate(f"fabric beat  {beat[-1][1]:.3f}", (beat[-1][0], beat[-1][1]),
                 textcoords="offset points", xytext=(-8, 8),
                 color=t["secondary"], fontsize=9, ha="right")

    # --- right: p50 vs N ---
    style_ax(axR, t, xs)
    axR.plot([r[0] for r in unit], [r[2] for r in unit], color=t["s_unit"], **lp)
    axR.plot([r[0] for r in beat], [r[2] for r in beat], color=t["s_beat"], **lp)
    axR.set_ylabel("p50  latency (ms)", color=t["secondary"], fontsize=10)
    axR.set_title("Cost per beat (we optimize stability, not speed)",
                  color=t["primary"], fontsize=12, fontweight="bold", loc="left")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=t["s_unit"], lw=2.5, marker="o", markersize=7,
               markeredgecolor=t["surface"], label="per-unit FFN compute  (M1-comparable)"),
        Line2D([0], [0], color=t["s_beat"], lw=2.5, marker="o", markersize=7,
               markeredgecolor=t["surface"], label="whole-fabric A-side beat  (NCCL + IPC fan-out)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=9, labelcolor=t["secondary"])

    fig.suptitle("MS-AFD M2 (dense) — fabric determinism vs micro-unit count  "
                 "(2× A100 80GB NVLink, Llama-3 8B FFN)",
                 color=t["primary"], fontsize=13, fontweight="bold", y=0.95)

    out = f"{out_dir}/m2_determinism{'' if mode == 'light' else '_dark'}.png"
    fig.savefig(out, facecolor=t["surface"], dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    unit_path = sys.argv[1] if len(sys.argv) > 1 else "results/m2/summary.csv"
    beat_path = sys.argv[2] if len(sys.argv) > 2 else "results/m2/summary_aside.csv"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "results/m2"
    for mode in ("light", "dark"):
        plot(unit_path, beat_path, out_dir, mode)


if __name__ == "__main__":
    main()
