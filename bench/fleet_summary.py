#!/usr/bin/env python3
"""M1 — summarize a fleet run: per-slice determinism + cross-slice fairness.

Reads every slice_*.csv (columns: iter,ms) in a directory and reports, per
slice, p50 / p99 / (p99:p50); then across the fleet:
  - p50 spread (max/min)  -> fairness: are all slices treated evenly?
  - worst per-slice p99:p50 -> the M1 determinism metric under load.

Pure stdlib. Usage: python bench/fleet_summary.py <dir>
"""

import csv
import glob
import os
import statistics
import sys


def pctile(sorted_vals, p):
    return sorted_vals[int(p * (len(sorted_vals) - 1))]


def load_ms(path):
    with open(path, newline="") as f:
        return [float(row["ms"]) for row in csv.DictReader(f)]


def slice_index(path):
    # slice_<k>.csv -> k, for stable ordering.
    base = os.path.basename(path)
    try:
        return int(base.split("_")[1].split(".")[0])
    except (IndexError, ValueError):
        return base


def main() -> None:
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    paths = sorted(glob.glob(os.path.join(d, "slice_*.csv")), key=slice_index)
    if not paths:
        print(f"no slice_*.csv found in {d}")
        return

    rows = []
    for p in paths:
        ms = sorted(load_ms(p))
        if not ms:
            print(f"  (empty: {os.path.basename(p)})")
            continue
        p50, p99 = pctile(ms, 0.50), pctile(ms, 0.99)
        rows.append((os.path.basename(p), len(ms), p50, p99, p99 / p50))

    if not rows:
        print("no non-empty slices")
        return

    print(f"{'slice':<16}{'n':>7}{'p50(ms)':>10}{'p99(ms)':>10}{'p99/p50':>10}")
    for name, n, p50, p99, r in rows:
        print(f"{name:<16}{n:>7}{p50:>10.4f}{p99:>10.4f}{r:>10.4f}")

    p50s = [r[2] for r in rows]
    ratios = [r[4] for r in rows]
    print("-" * 53)
    print(f"slices              : {len(rows)}")
    print(f"p50 across slices   : min {min(p50s):.4f}  "
          f"median {statistics.median(p50s):.4f}  max {max(p50s):.4f}")
    print(f"p50 spread (max/min): {max(p50s) / min(p50s):.3f}"
          f"   <-- cross-slice fairness (1.0 = perfectly even)")
    print(f"p99/p50 per slice   : min {min(ratios):.4f}  "
          f"median {statistics.median(ratios):.4f}  max {max(ratios):.4f}")
    print(f"worst p99/p50       : {max(ratios):.4f}"
          f"   <-- M1 determinism metric (want ~1.1, flat as N grows)")


if __name__ == "__main__":
    main()
