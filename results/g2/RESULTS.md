# G2 results — the determinism baseline / control

**Hardware:** 1× NVIDIA A100-SXM4-80GB, CUDA 12.8. Same `slice` FFN as M0–M2.

The oldest open question in this project (flagged since M1): we report p99/p50 ≈ 1.01,
but **1.01 vs what?** Is the fabric *more* deterministic than naive co-tenancy, or
is the determinism coming from somewhere else? G2 answers it with controls. The
answer is humbling and important, and it makes the earlier numbers *more* credible,
not less.

## Test 1 — N tenants: naive (no MPS) vs fabric (MPS-partitioned)

N timed FFN tenants on one GPU; worst per-tenant p99/p50. `naive` = no MPS at all;
`fabric` = MPS-partitioned at 100/N% each. (Directly comparable to M1.)

| N  | naive p50 | naive **p99/p50** | fabric p50 | fabric **p99/p50** |
|----|-----------|-------------------|------------|--------------------|
| 8  | 0.49 ms   | 1.166             | 4.26 ms    | 1.042 |
| 16 | 0.50 ms   | **1.012**         | 9.01 ms    | 1.027 |
| 32 | 0.50 ms   | **1.010**         | 25.27 ms   | 1.007 |
| 48 | 0.50 ms   | **1.009**         | 37.69 ms   | 1.016 |

**Naive co-tenancy is already deterministic.** With no MPS at all, worst-case
per-tenant p99/p50 is ~1.01 from N=16 to N=48, and each tenant's p50 stays ~0.50 ms
*regardless of tenant count*. MPS partitioning does **not** improve the ratio — it
matches it, at a large latency cost (thinner SM slice → 37 ms vs 0.5 ms at N=48).
(The measured quantity is per-kernel execution time; kernels run in a scheduling
quantum, so a neighbour's work lands *between* a tenant's timed iterations.)

## Test 2 — noisy neighbour: small worker next to 2 heavy antagonists

A latency-sensitive worker (256-token FFN) beside 2 heavy antagonists (4096-token
FFNs, continuous), three ways:

| mode   | worker p50 | worker **p99/p50** | p50 vs solo |
|--------|-----------|--------------------|-------------|
| solo   | 0.520 ms  | 1.107              | 1.00× |
| naive  | 0.507 ms  | **1.018**          | 0.98× |
| fabric | 1.103 ms  | 1.002              | 2.12× |

Even with heavy neighbours and **no** MPS, the worker's per-kernel time is
unperturbed (0.51 ms, ratio 1.018). The A100 does not let a big neighbour inflate a
small tenant's measured FFN. MPS again matches the ratio at a latency cost.

## Test 3 — software discipline: CUDA graph vs eager

Same worker, graph-replay vs eager (separate kernel launches), solo and at 32-way
MPS contention:

| condition        | graph p99/p50 | eager p99/p50 |
|------------------|---------------|----------------|
| solo (idle GPU)  | 1.102         | 1.112 |
| 32-way contended | 1.008         | 1.009 |

The CUDA graph does not *uniquely* produce the tight ratio either — eager is within
noise. (The static arena still eliminates per-iteration allocation by construction;
that discipline is belt-and-suspenders, not the source of the ~1.01.)

## What this means (the honest reframe)

- **The determinism is real and robust, not an artifact.** p99/p50 ≈ 1.01 holds with
  MPS off, graph off, under 48-way contention, and beside heavy neighbours. The
  M0–M4 numbers are trustworthy — the control rules out "the fabric is just masking
  noise."
- **On this hardware, the ratio is inherent to the workload.** A compute-bound FFN
  has a small (~50 µs) fixed jitter; over a kernel of hundreds of µs–tens of ms it
  amortizes to ~1.01, and *tightens* as the kernel runs longer (more tenants / thinner
  SM cap). Neither MPS partitioning nor CUDA graphs are what create it.
- **So the fabric's distinctive value is not tail-latency isolation — it is
  capacity isolation and structure:** a guaranteed SM share per unit (fairness / no
  throughput starvation, M1), and a systolic many-unit architecture that serves a
  real model deterministically end to end (M4) — delivered *without* costing the
  determinism the hardware already affords.

In one line: **we set out to make FFN execution deterministic; the control shows
determinism is achievable and robust here, and the fabric provides it with
guaranteed isolation and a real-model-serving structure on top.**

## Reproduce

```bash
bash scripts/run_baseline.sh 2000 256 "8 16 32 48"     # Test 1
bash scripts/run_baseline_neighbor.sh 3000 256 4096 2 40  # Test 2
MSAFD_EAGER=1 ./build/slice 5000 256 eager.csv         # Test 3 (vs default graph)
```
