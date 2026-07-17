#!/usr/bin/env python3
"""M1 — aggregate per-slice latency into a determinism-vs-slice-count summary.

Reads every latency_rank*.csv in a directory (one per concurrent slice), computes
each slice's p50/p99/ratio, then reports the cross-slice distribution — the M1
question being "does per-slice determinism hold as slice count rises?".

Usage: aggregate.py OUTDIR MODE N PCT [SUMMARY_CSV]
  Prints a per-run summary and, if SUMMARY_CSV is given, appends one row to it.
"""

import csv
import glob
import os
import sys


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = p / 100.0 * (len(s) - 1)
    lo = int(idx)
    frac = idx - lo
    if lo + 1 >= len(s):
        return s[-1]
    return s[lo] * (1 - frac) + s[lo + 1] * frac


def load(path):
    with open(path, newline="") as fh:
        return [float(r["ms"]) for r in csv.DictReader(fh)]


def main():
    outdir = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "?"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    pct = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    summary_csv = sys.argv[5] if len(sys.argv) > 5 else None

    files = sorted(glob.glob(os.path.join(outdir, "latency_rank*.csv")))
    per_slice = []  # (rank, p50, p99, ratio)
    for f in files:
        ms = load(f)
        if not ms:
            continue
        p50 = percentile(ms, 50)
        p99 = percentile(ms, 99)
        per_slice.append((f, p50, p99, p99 / p50))

    if not per_slice:
        print(f"[aggregate] no data in {outdir}", file=sys.stderr)
        sys.exit(1)

    p50s = [s[1] for s in per_slice]
    p99s = [s[2] for s in per_slice]
    ratios = [s[3] for s in per_slice]

    p50_med = percentile(p50s, 50)
    p99_worst = max(p99s)
    ratio_med = percentile(ratios, 50)
    ratio_worst = max(ratios)

    print(f"\n=== M1 {mode}  N={n}  pct_each={pct}%  ({len(per_slice)} slices) ===")
    print(f"  per-slice p50   : min {min(p50s):.3f}  median {p50_med:.3f}  max {max(p50s):.3f} ms")
    print(f"  per-slice p99   : min {min(p99s):.3f}  median {percentile(p99s,50):.3f}  max {p99_worst:.3f} ms")
    print(f"  per-slice p99/p50: min {min(ratios):.4f}  median {ratio_med:.4f}  WORST {ratio_worst:.4f}")

    if summary_csv:
        new = not os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["mode", "slices", "pct_each", "slice_p50_ms_median",
                            "slice_p99_ms_worst", "ratio_median", "ratio_worst"])
            w.writerow([mode, n, pct, f"{p50_med:.4f}", f"{p99_worst:.4f}",
                        f"{ratio_med:.4f}", f"{ratio_worst:.4f}"])


if __name__ == "__main__":
    main()
