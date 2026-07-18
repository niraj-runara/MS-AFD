// MS-AFD · M3-full Rungs 2-3 — the MoE-FFN op HF calls during generation.
//
// A C-ABI shared library. HF runs attention/KV/router/generation; for each MoE
// layer it hands us the hidden states + router top-k, and we compute that layer's
// MoE output with the fabric's own FFN math (the exact GEMMs + SwiGLU proven in
// ffn.cu / m3_layer to reproduce HF to ~0.004 rel-L2), capacity-padded per expert.
//
// Weights are consumed in HF's FUSED layout directly (no copy): gate_up_proj
// [E, 2F, D] and down_proj [E, D, F], both [out, in] like nn.Linear. We use a
// transpose-flag GEMM (OP_T) so no pre-transpose / extra memory is needed.
//
// extern "C" so Python can call it via ctypes with torch .data_ptr()s.

#include <cstdio>
#include <cstdlib>
#include <vector>
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

__global__ void swiglu_k(const bf16* gate, const bf16* up, bf16* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        out[i] = __float2bfloat16((g / (1.0f + __expf(-g))) * u);
    }
}

// out[i] += w * src[i]   (fp32 accumulate)
__global__ void waxpy_k(bf16* out, const bf16* src, float w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2bfloat16(__bfloat162float(out[i]) + w * __bfloat162float(src[i]));
}

extern "C" void msafd_moe_ffn(
    const void* hidden,     // [T, D] bf16, device
    const void* gate_up,    // [E, 2F, D] bf16, device (HF fused)
    const void* down,       // [E, D, F] bf16, device (HF)
    const int* topk_idx,    // [T, topk] int32, HOST
    const float* topk_w,    // [T, topk] float32, HOST
    void* out,              // [T, D] bf16, device (written)
    int T, int D, int F, int E, int topk) {

    static cublasHandle_t handle = nullptr;
    static cudaStream_t stream = nullptr;
    if (!handle) { CUBLAS_CHECK(cublasCreate(&handle)); CUDA_CHECK(cudaStreamCreate(&stream));
                   CUBLAS_CHECK(cublasSetStream(handle, stream)); }

    const auto* Hd = static_cast<const bf16*>(hidden);
    const auto* GU = static_cast<const bf16*>(gate_up);
    const auto* DN = static_cast<const bf16*>(down);
    auto* Out = static_cast<bf16*>(out);

    // Group tokens by expert (host routing); capacity C = max per-expert load.
    std::vector<std::vector<int>> toks(E);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < topk; ++k)
            toks[topk_idx[(size_t)t * topk + k]].push_back(t);
    int C = 1;
    for (auto& v : toks) if ((int)v.size() > C) C = (int)v.size();

    // Per-call scratch (correctness demo; not perf-critical).
    bf16 *tin, *g, *u, *hid, *tout;
    CUDA_CHECK(cudaMalloc(&tin,  (size_t)C * D * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g,    (size_t)C * F * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&u,    (size_t)C * F * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&hid,  (size_t)C * F * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&tout, (size_t)C * D * sizeof(bf16)));

    CUDA_CHECK(cudaMemsetAsync(Out, 0, (size_t)T * D * sizeof(bf16), stream));

    const size_t guStride = (size_t)2 * F * D;   // per-expert gate_up block
    const size_t dnStride = (size_t)D * F;        // per-expert down block

    for (int e = 0; e < E; ++e) {
        if (toks[e].empty()) continue;
        const int load = (int)toks[e].size();

        // Build the capacity tile: real tokens in slots [0,load), zero-pad rest.
        CUDA_CHECK(cudaMemsetAsync(tin, 0, (size_t)C * D * sizeof(bf16), stream));
        for (int s = 0; s < load; ++s)
            CUDA_CHECK(cudaMemcpyAsync(tin + (size_t)s * D, Hd + (size_t)toks[e][s] * D,
                                       (size_t)D * sizeof(bf16), cudaMemcpyDeviceToDevice, stream));

        const bf16* gate_hf = GU + (size_t)e * guStride;                 // [F, D]
        const bf16* up_hf   = GU + (size_t)e * guStride + (size_t)F * D;  // [F, D]
        const bf16* down_hf = DN + (size_t)e * dnStride;                 // [D, F]

        gemm_linear(handle, tin, gate_hf, g, C, D, F);   // gate = tin @ gate^T  -> [C,F]
        gemm_linear(handle, tin, up_hf,   u, C, D, F);   // up   = tin @ up^T    -> [C,F]
        int n = C * F, thr = 256, blk = (n + thr - 1) / thr;
        swiglu_k<<<blk, thr, 0, stream>>>(g, u, hid, n);
        gemm_linear(handle, hid, down_hf, tout, C, F, D); // out = hid @ down^T  -> [C,D]

        // Scatter-combine into per-token output, weighted by the router.
        int dblk = (D + thr - 1) / thr;
        for (int s = 0; s < load; ++s) {
            int t = toks[e][s];
            float w = 0.0f;
            for (int k = 0; k < topk; ++k)
                if (topk_idx[(size_t)t * topk + k] == e) { w = topk_w[(size_t)t * topk + k]; break; }
            waxpy_k<<<dblk, thr, 0, stream>>>(Out + (size_t)t * D, tout + (size_t)s * D, w, D);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(tin)); CUDA_CHECK(cudaFree(g)); CUDA_CHECK(cudaFree(u));
    CUDA_CHECK(cudaFree(hid)); CUDA_CHECK(cudaFree(tout));
}
