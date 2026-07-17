#!/usr/bin/env python3
"""M0 · Step 6 — summarize per-iteration latency from the persistent loop.

Reads latency.csv (columns: iter,ms) produced by slice.cu and reports the
determinism metric: p50, p99, and the p99/p50 ratio. Pure stdlib, no deps.

Usage:
    python bench/latency.py [latency.csv]
"""

import csv
import statistics
import sys


def percentile(sorted_vals, p):
    """Nearest-rank percentile on an already-sorted list; p in [0, 1]."""
    if not sorted_vals:
        raise ValueError("no samples")
    idx = int(p * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "latency.csv"
    with open(path, newline="") as f:
        ms = [float(row["ms"]) for row in csv.DictReader(f)]

    if not ms:
        print(f"{path}: no samples")
        return

    ms_sorted = sorted(ms)
    p50 = percentile(ms_sorted, 0.50)
    p90 = percentile(ms_sorted, 0.90)
    p99 = percentile(ms_sorted, 0.99)
    p999 = percentile(ms_sorted, 0.999)
    mean = statistics.fmean(ms)
    std = statistics.pstdev(ms)

    print(f"samples   : {len(ms)}")
    print(f"min / max : {ms_sorted[0]:.4f} / {ms_sorted[-1]:.4f} ms")
    print(f"mean +- sd: {mean:.4f} +- {std:.4f} ms")
    print(f"p50       : {p50:.4f} ms")
    print(f"p90       : {p90:.4f} ms")
    print(f"p99       : {p99:.4f} ms")
    print(f"p99.9     : {p999:.4f} ms")
    print(f"p99/p50   : {p99 / p50:.4f}   <-- determinism metric")


if __name__ == "__main__":
    main()
