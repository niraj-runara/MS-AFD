#pragma once
// M0 · Step 3 — FFN shape and weights (one unit of FFN work).
//
// One Llama-3 8B dense FFN block: SwiGLU = down( silu(gate(x)) * up(x) ).
// Weights live in the Arena (HBM). fp16 storage, fp32 accumulation on A100
// tensor cores. Header-only (included by exactly one .cu).

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "arena.h"

namespace msafd {

#define MSAFD_CUBLAS_CHECK(expr)                                               \
    do {                                                                      \
        cublasStatus_t _s = (expr);                                          \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "[cublas] %s failed at %s:%d (status %d)\n", \
                         #expr, __FILE__, __LINE__, (int)_s);                \
            std::abort();                                                   \
        }                                                                   \
    } while (0)

struct FfnConfig {
    int d_model        = 4096;   // Llama-3 8B hidden size
    int d_intermediate = 14336;  // Llama-3 8B FFN intermediate size
    int tokens         = 256;    // token batch processed per slice iteration
};

// h[i] = silu(gate[i]) * up[i], elementwise, fp16 in/out.
__global__ void swiglu_kernel(const __half* __restrict__ gate,
                              const __half* __restrict__ up,
                              __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        float s = g / (1.0f + expf(-g));  // SiLU / swish
        out[i]  = __float2half(s * u);
    }
}

class Ffn {
public:
    Ffn(const FfnConfig& cfg, Arena& arena) : cfg_(cfg) {
        MSAFD_CUBLAS_CHECK(cublasCreate(&handle_));
        MSAFD_CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));

        const std::size_t dm = cfg_.d_model, di = cfg_.d_intermediate,
                          T = cfg_.tokens;

        // Weights (row-major logical layout, PyTorch nn.Linear convention).
        Wg_ = (__half*)arena.alloc(sizeof(__half) * di * dm, "w_gate");
        Wu_ = (__half*)arena.alloc(sizeof(__half) * di * dm, "w_up");
        Wd_ = (__half*)arena.alloc(sizeof(__half) * dm * di, "w_down");

        // Intermediate activations.
        gate_ = (__half*)arena.alloc(sizeof(__half) * T * di, "act_gate");
        up_   = (__half*)arena.alloc(sizeof(__half) * T * di, "act_up");
        h_    = (__half*)arena.alloc(sizeof(__half) * T * di, "act_swiglu");

        // Fixed cuBLAS workspace, so no allocation happens during graph capture.
        ws_bytes_ = 32ull << 20;
        ws_       = arena.alloc(ws_bytes_, "cublas_ws");
        MSAFD_CUBLAS_CHECK(cublasSetWorkspace(handle_, ws_, ws_bytes_));

        init_weights();
    }
    ~Ffn() { cublasDestroy(handle_); }
    Ffn(const Ffn&) = delete;
    Ffn& operator=(const Ffn&) = delete;

    // Eager (and capture-safe) forward:
    //   gate = X @ Wg^T ; up = X @ Wu^T ; h = silu(gate)*up ; out = h @ Wd^T
    void forward(const __half* input, __half* output, cudaStream_t s) {
        const int dm = cfg_.d_model, di = cfg_.d_intermediate, T = cfg_.tokens;
        MSAFD_CUBLAS_CHECK(cublasSetStream(handle_, s));
        linear(input, Wg_, gate_, T, di, dm);  // [T,di]
        linear(input, Wu_, up_, T, di, dm);    // [T,di]
        int n = T * di, block = 256, grid = (n + block - 1) / block;
        swiglu_kernel<<<grid, block, 0, s>>>(gate_, up_, h_, n);
        linear(h_, Wd_, output, T, dm, di);    // [T,dm]
    }

    const FfnConfig&              config() const { return cfg_; }
    const std::vector<__half>&    host_wg() const { return h_wg_; }
    const std::vector<__half>&    host_wu() const { return h_wu_; }
    const std::vector<__half>&    host_wd() const { return h_wd_; }

private:
    // Y[T,N] = X[T,K] @ W[N,K]^T, all row-major logical layout.
    // cuBLAS is column-major, so we compute Y^T[N,T] = W[N,K] @ X[K,T]:
    //   op(A)=W^T  (W stored col-major [K,N], transa=T)  -> [N,K]
    //   op(B)=X    (X stored col-major [K,T], transb=N)  -> [K,T]
    //   C = Y^T stored col-major [N,T] == Y row-major [T,N]
    void linear(const __half* X, const __half* W, __half* Y, int T, int N,
                int K) {
        const float alpha = 1.0f, beta = 0.0f;
        MSAFD_CUBLAS_CHECK(cublasGemmEx(
            handle_, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &alpha, W, CUDA_R_16F,
            K, X, CUDA_R_16F, K, &beta, Y, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Deterministic small pseudo-random fill in ~[-scale, scale], kept on the
    // host too so slice.cu can compute a CPU reference for correctness.
    void init_weights() {
        const std::size_t dm = cfg_.d_model, di = cfg_.d_intermediate;
        h_wg_.resize(di * dm);
        h_wu_.resize(di * dm);
        h_wd_.resize(dm * di);
        fill(h_wg_, 1u, 0.02f);
        fill(h_wu_, 2u, 0.02f);
        fill(h_wd_, 3u, 0.02f);
        MSAFD_CUDA_CHECK(cudaMemcpy(Wg_, h_wg_.data(),
                                    h_wg_.size() * sizeof(__half),
                                    cudaMemcpyHostToDevice));
        MSAFD_CUDA_CHECK(cudaMemcpy(Wu_, h_wu_.data(),
                                    h_wu_.size() * sizeof(__half),
                                    cudaMemcpyHostToDevice));
        MSAFD_CUDA_CHECK(cudaMemcpy(Wd_, h_wd_.data(),
                                    h_wd_.size() * sizeof(__half),
                                    cudaMemcpyHostToDevice));
    }

    static void fill(std::vector<__half>& v, unsigned seed, float scale) {
        for (std::size_t i = 0; i < v.size(); ++i) {
            unsigned x = (unsigned)(i * 2654435761u + seed * 40503u);
            float    r = ((x >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
            v[i]       = __float2half(r * scale);
        }
    }

    FfnConfig      cfg_;
    cublasHandle_t handle_ = nullptr;
    __half*        Wg_ = nullptr;
    __half*        Wu_ = nullptr;
    __half*        Wd_ = nullptr;
    __half*        gate_ = nullptr;
    __half*        up_ = nullptr;
    __half*        h_ = nullptr;
    void*          ws_ = nullptr;
    std::size_t    ws_bytes_ = 0;
    std::vector<__half> h_wg_, h_wu_, h_wd_;  // host mirror for CPU reference
};

}  // namespace msafd
