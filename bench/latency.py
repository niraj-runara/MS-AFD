#!/usr/bin/env python3
"""M0 · Step 6 — summarize per-iteration latency from the persistent loop.

Reads latency.csv (one elapsed-ms-per-iteration, produced by slice.cu) and
reports the M0 determinism metric: p50, p99, and the p99/p50 ratio.

Pure stdlib — no numpy dependency.

Usage: python bench/latency.py [latency.csv]
"""

import csv
import sys


def percentile(values, p):
    """Linear-interpolation percentile over a sorted list (p in [0, 100])."""
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


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "latency.csv"
    try:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            ms = [float(row["ms"]) for row in reader]
    except FileNotFoundError:
        print(f"error: {path} not found — run ./build/slice first", file=sys.stderr)
        sys.exit(1)

    if not ms:
        print(f"error: no rows in {path}", file=sys.stderr)
        sys.exit(1)

    n = len(ms)
    mean = sum(ms) / n
    var = sum((x - mean) ** 2 for x in ms) / n
    std = var ** 0.5
    p50 = percentile(ms, 50)
    p90 = percentile(ms, 90)
    p99 = percentile(ms, 99)
    p999 = percentile(ms, 99.9)

    print(f"latency summary  ({n} iterations, {path})")
    print(f"  min    {min(ms):9.4f} ms")
    print(f"  p50    {p50:9.4f} ms")
    print(f"  p90    {p90:9.4f} ms")
    print(f"  p99    {p99:9.4f} ms")
    print(f"  p99.9  {p999:9.4f} ms")
    print(f"  max    {max(ms):9.4f} ms")
    print(f"  mean   {mean:9.4f} ms")
    print(f"  std    {std:9.4f} ms")
    print(f"  ---")
    print(f"  p99/p50 = {p99 / p50:.3f}   <-- M0 determinism metric")


if __name__ == "__main__":
    main()
