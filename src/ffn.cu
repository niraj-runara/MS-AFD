// M0 · Step 3 — FFN implementation: 3 bf16 GEMMs + a SwiGLU activation kernel.
//
// cuBLAS is column-major; our tensors are row-major. For a row-major product
// C[M,N] = A[M,K] @ B[K,N] we ask cuBLAS for the column-major C^T = B^T @ A^T,
// which is the same bytes, via:
//   gemm(N, M, K, B(ld=N), A(ld=K), C(ld=N))   with OP_N/OP_N.
// No explicit transposes, no data movement.

#include "ffn.h"
#include "arena.h"
#include "check.h"

#include <cmath>
#include <random>
#include <vector>
#include <cuda_bf16.h>

namespace msafd {

// h = silu(gate) * up, elementwise. silu(x) = x * sigmoid(x). Compute in fp32.
__global__ void swiglu_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                              __nv_bfloat16* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float s = g / (1.0f + __expf(-g));  // silu
        out[i] = __float2bfloat16(s * u);
    }
}

// Row-major C[M,N] = A[M,K] @ B[K,N], bf16 in/out, fp32 accumulate.
static void gemm_rm(cublasHandle_t h, const __nv_bfloat16* A,
                    const __nv_bfloat16* B, __nv_bfloat16* C, int M, int K,
                    int N) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                              CUDA_R_16BF, N, A, CUDA_R_16BF, K, &beta, C,
                              CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
}

// Quantize a host fp32 buffer to bf16 and upload it to a device region.
static void upload_bf16(void* dst, const std::vector<float>& src) {
    std::vector<__nv_bfloat16> tmp(src.size());
    for (size_t i = 0; i < src.size(); ++i) tmp[i] = __float2bfloat16(src[i]);
    CUDA_CHECK(cudaMemcpy(dst, tmp.data(), tmp.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
}

Ffn::Ffn(const FfnConfig& cfg, Arena& arena) : cfg_(cfg) {
    const int D = cfg.d_model, F = cfg.d_intermediate, T = cfg.tokens;
    const size_t bf = sizeof(__nv_bfloat16);

    // Weights (bf16) — the ~9 MB served from HBM every beat.
    wg_ = arena.alloc((size_t)D * F * bf, "wg");
    wu_ = arena.alloc((size_t)D * F * bf, "wu");
    wd_ = arena.alloc((size_t)F * D * bf, "wd");

    // Scratch activations (bf16).
    gate_   = arena.alloc((size_t)T * F * bf, "gate");
    up_     = arena.alloc((size_t)T * F * bf, "up");
    hidden_ = arena.alloc((size_t)T * F * bf, "hidden");

    // Fixed-seed init; small scale keeps activations numerically tame so the
    // fp32-vs-bf16 correctness gate has a clean margin.
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    wg_h_.resize((size_t)D * F);
    wu_h_.resize((size_t)D * F);
    wd_h_.resize((size_t)F * D);
    for (auto& v : wg_h_) v = dist(rng);
    for (auto& v : wu_h_) v = dist(rng);
    for (auto& v : wd_h_) v = dist(rng);

    upload_bf16(wg_, wg_h_);
    upload_bf16(wu_, wu_h_);
    upload_bf16(wd_, wd_h_);
}

void Ffn::forward(const void* input, void* output, cublasHandle_t handle,
                  cudaStream_t stream) {
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const auto* in  = static_cast<const __nv_bfloat16*>(input);
    auto*       out = static_cast<__nv_bfloat16*>(output);
    const int   T = cfg_.tokens, D = cfg_.d_model, F = cfg_.d_intermediate;

    // gate = in @ Wg ; up = in @ Wu   ([T,D] @ [D,F] -> [T,F])
    gemm_rm(handle, in, static_cast<const __nv_bfloat16*>(wg_),
            static_cast<__nv_bfloat16*>(gate_), T, D, F);
    gemm_rm(handle, in, static_cast<const __nv_bfloat16*>(wu_),
            static_cast<__nv_bfloat16*>(up_), T, D, F);

    // hidden = silu(gate) * up
    const int n = T * F;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_),
        static_cast<const __nv_bfloat16*>(up_),
        static_cast<__nv_bfloat16*>(hidden_), n);

    // out = hidden @ Wd   ([T,F] @ [F,D] -> [T,D])
    gemm_rm(handle, static_cast<const __nv_bfloat16*>(hidden_),
            static_cast<const __nv_bfloat16*>(wd_), out, T, F, D);
}

void Ffn::reference(const float* in, float* out) const {
    const int T = cfg_.tokens, D = cfg_.d_model, F = cfg_.d_intermediate;
    std::vector<float> h((size_t)T * F);

    for (int m = 0; m < T; ++m) {
        for (int k = 0; k < F; ++k) {
            float ga = 0.0f, ua = 0.0f;
            for (int j = 0; j < D; ++j) {
                float x = in[(size_t)m * D + j];
                ga += x * wg_h_[(size_t)j * F + k];
                ua += x * wu_h_[(size_t)j * F + k];
            }
            float s = ga / (1.0f + std::exp(-ga));  // silu
            h[(size_t)m * F + k] = s * ua;
        }
    }
    for (int m = 0; m < T; ++m) {
        for (int nn = 0; nn < D; ++nn) {
            float acc = 0.0f;
            for (int k = 0; k < F; ++k)
                acc += h[(size_t)m * F + k] * wd_h_[(size_t)k * D + nn];
            out[(size_t)m * D + nn] = acc;
        }
    }
}

}  // namespace msafd
