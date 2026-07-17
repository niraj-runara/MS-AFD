# M0 results — single MPS slice

**Hardware:** 1× NVIDIA A100 80GB PCIe (RunPod), driver 550.90.12, CUDA 12.8, Ubuntu 24.04.
**Model:** Llama-3 8B dense FFN — SwiGLU, d_model=4096, d_intermediate=14336, fp16 storage / fp32 accumulation.
**Config:** tokens=256, 5000 timed iterations, MPS at `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10`.

## Determinism (the M0 headline)

| metric | value |
|--------|-------|
| min    | 4.4881 ms |
| p50    | 4.4932 ms |
| p90    | 4.4954 ms |
| p99    | 4.4998 ms |
| p99.9  | 4.5998 ms |
| max    | 4.6981 ms |
| mean   | 4.4936 ms |
| std    | 0.0065 ms |
| **p99/p50** | **1.001** |

Raw per-iteration timings: [latency_mps10.csv](latency_mps10.csv) (5000 rows).

## Mechanisms verified

- **A — MPS compute isolation:** 10% SM cap took effect (full-GPU p50 ≈ 0.57 ms → 4.49 ms at 10%, ~8× slower, determinism tightened to p99/p50=1.001).
- **C — static 1 GB arena:** exactly one `cudaMalloc`; fixed-offset, bounds-checked allocator; 412 MB / 1 GB used.
- **D — CUDA Graph:** FFN captured into a graph; graph output **bit-exact** vs eager forward.
- **Correctness:** GPU output vs CPU fp32 reference (token 0, 8 channels) max rel err 0.0001; full output finite (no NaN/inf).

## Reproduce

```bash
# on the A100 box
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
nvidia-cuda-mps-control -d
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80 && cmake --build build -j
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10 ./build/slice 5000 256
python3 bench/latency.py latency.csv
```
