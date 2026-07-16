#!/usr/bin/env bash
# M0 · Step 1 — launch NVIDIA MPS and pin this slice's SM share.
# Boilerplate: adjust device and thread percentage, then run ./build/slice.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log

# Slice compute budget: fraction of the GPU's SMs for this process.
# Start at 10% for M0; sweep this in M1.
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10

# Start the MPS control daemon (no-op if already running).
nvidia-cuda-mps-control -d

echo "MPS up. CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE}"
echo "Now run: ./build/slice"

# To stop MPS later:
#   echo quit | nvidia-cuda-mps-control
