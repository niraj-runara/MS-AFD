// MS-AFD · M3 (dense) — the dense-FFN op HF calls during generation.
//
// A C-ABI shared library. HF runs the real Llama-3 8B (attention, KV cache,
// generation); for every decoder layer it hands us that layer's post-norm hidden
// states + the three MLP weight matrices, and we compute the SwiGLU FFN with the
// fabric's own GEMM+SwiGLU math (the same as ffn.h / the M2 micro-unit). A dense
// model has no router — every token goes through the one FFN — so this is niraj's
// MoE op minus routing / capacity padding / scatter-combine.
//
// Weights are consumed in HF's nn.Linear layout directly (no copy): gate/up are
// [F, D] and down is [D, F], both [out, in]. A transpose-flag GEMM (OP_T) needs
// no pre-transpose. bf16 storage / fp32 accumulate, to match HF's native dtype.
//
// extern "C" so Python can call it via ctypes with torch .data_ptr()s.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include "check.h"

using bf16 = __nv_bfloat16;

// C[M,N] = A[M,K] @ Bhf[N,K]^T   (Bhf is a Linear-style [out=N, in=K] weight).
static void gemm_linear(cublasHandle_t h, const bf16* A, const bf16* Bhf, bf16* C,
                        int M, int K, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                              Bhf, CUDA_R_16BF, K, A, CUDA_R_16BF, K, &beta,
                              C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
}

// out[i] = silu(gate[i]) * up[i]   (fp32 math, matches HF's SiLU MLP)
__global__ void swiglu_k(const bf16* gate, const bf16* up, bf16* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        out[i] = __float2bfloat16((g / (1.0f + __expf(-g))) * u);
    }
}

// out[T,D] = down( silu(gate(hidden)) * up(hidden) ), for all T tokens.
//   hidden  [T, D] bf16 device
//   gate_w  [F, D] bf16 device (HF gate_proj.weight)
//   up_w    [F, D] bf16 device (HF up_proj.weight)
//   down_w  [D, F] bf16 device (HF down_proj.weight)
//   out     [T, D] bf16 device (written)
extern "C" void msafd_dense_ffn(const void* hidden, const void* gate_w,
                                const void* up_w, const void* down_w, void* out,
                                int T, int D, int F) {
    static cublasHandle_t handle = nullptr;
    static cudaStream_t   stream = nullptr;
    if (!handle) {
        CUBLAS_CHECK(cublasCreate(&handle));
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasSetStream(handle, stream));
    }

    // Scratch for the two [T,F] projections + the SwiGLU product; grows on demand
    // and persists across calls (avoids malloc/free churn during generation).
    static bf16*  scratch = nullptr;
    static size_t scratch_cap = 0;       // in elements, per [T,F] band
    const size_t band = (size_t)T * F;
    if (band > scratch_cap) {
        if (scratch) CUDA_CHECK(cudaFree(scratch));
        CUDA_CHECK(cudaMalloc(&scratch, 3 * band * sizeof(bf16)));
        scratch_cap = band;
    }
    bf16* g = scratch;
    bf16* u = scratch + band;
    bf16* h = scratch + 2 * band;

    const auto* H  = static_cast<const bf16*>(hidden);
    const auto* Wg = static_cast<const bf16*>(gate_w);
    const auto* Wu = static_cast<const bf16*>(up_w);
    const auto* Wd = static_cast<const bf16*>(down_w);
    auto*       O  = static_cast<bf16*>(out);

    gemm_linear(handle, H, Wg, g, T, D, F);   // gate = hidden @ gate_w^T -> [T,F]
    gemm_linear(handle, H, Wu, u, T, D, F);   // up   = hidden @ up_w^T   -> [T,F]
    int n = (int)band, thr = 256, blk = (n + thr - 1) / thr;
    swiglu_k<<<blk, thr, 0, stream>>>(g, u, h, n);
    gemm_linear(handle, h, Wd, O, T, F, D);   // out  = h @ down_w^T      -> [T,D]

    CUDA_CHECK(cudaStreamSynchronize(stream));
}
