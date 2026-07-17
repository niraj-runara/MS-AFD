#pragma once
// M0 · Step 3 — FFN shape and weights (one unit of FFN work).
//
// Represents a single FFN (one MoE expert): SwiGLU = gate & up projections
// then a down projection. Weights live in the Arena (HBM).
//
// Dims = one Qwen3-30B-A3B expert (fine-grained MoE): hidden_size 2048,
// moe_intermediate_size 768, SiLU/SwiGLU, bf16. One expert = one micro-unit,
// ~9 MB of weights (3 x 2048 x 768 x 2 bytes) served from HBM.

#include <cstddef>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace msafd {

class Arena;

struct FfnConfig {
    // One Qwen3-30B-A3B expert.
    int d_model        = 2048;   // hidden_size
    int d_intermediate = 768;    // moe_intermediate_size (per-expert FFN width)

    // Fixed token tile processed per systolic beat (per graph launch). We don't
    // care about latency; a fixed M keeps per-iter timing stable and the GEMM
    // non-trivial. Tunable knob for M0/M1.
    int tokens = 256;

    // Weights and activations are bf16 (CUDA_R_16BF); GEMMs accumulate in fp32
    // (CUBLAS_COMPUTE_32F).
};

class Ffn {
public:
    // Reserve weight + scratch regions in the arena, fill weights with a fixed
    // random seed, and upload them (bf16) to HBM.
    Ffn(const FfnConfig& cfg, Arena& arena);

    // Eager/graph forward on `stream` using `handle`:
    //   input [tokens, d_model] bf16  ->  gate/up GEMM -> SwiGLU -> down GEMM
    //   -> output [tokens, d_model] bf16.
    // All buffers are device pointers. Contains no host sync or allocation, so
    // it is safe to call inside a CUDA Graph capture.
    void forward(const void* input, void* output, cublasHandle_t handle,
                 cudaStream_t stream);

    // CPU fp32 reference of forward() for the one-time M0 correctness gate.
    // in/out are host [tokens, d_model] fp32.
    void reference(const float* in, float* out) const;

    const FfnConfig& config() const { return cfg_; }

private:
    FfnConfig cfg_;

    // Weights in the arena (bf16). Row-major:
    //   wg_, wu_: [d_model, d_intermediate]   wd_: [d_intermediate, d_model]
    void* wg_ = nullptr;
    void* wu_ = nullptr;
    void* wd_ = nullptr;

    // Scratch in the arena (bf16): [tokens, d_intermediate].
    void* gate_   = nullptr;
    void* up_     = nullptr;
    void* hidden_ = nullptr;

    // Host fp32 copies of the weights, kept only for reference().
    std::vector<float> wg_h_, wu_h_, wd_h_;
};

}  // namespace msafd
