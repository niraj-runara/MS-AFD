#pragma once
// M0 · Step 3 — FFN shape and weights (one unit of FFN work).
//
// Represents a single FFN (or one MoE expert): SwiGLU = gate & up projections
// then down projection. Weights live in the Arena (HBM). Boilerplate only.

#include <cstddef>

namespace msafd {

struct FfnConfig {
    // TODO: set real dims for M0. Placeholder = Mixtral-expert-sized, SwiGLU, fp16.
    int d_model      = 4096;
    int d_intermediate = 14336;
    // dtype, batch/token count, etc. TODO.
};

class Ffn {
public:
    // Bind weight pointers inside the arena; fill for M0 (random is fine). TODO.
    explicit Ffn(const FfnConfig& cfg /*, Arena& arena */);

    // Eager forward: input -> gate/up GEMM -> SwiGLU -> down GEMM -> output.
    // Step 3 runs this once for correctness before it is captured into a graph.
    // TODO: cuBLAS GEMMs + activation kernel on `stream`.
    void forward(const void* input, void* output, void* stream);

private:
    FfnConfig cfg_;
    // TODO: device pointers for gate/up/down weights (offsets into the arena).
};

}  // namespace msafd
