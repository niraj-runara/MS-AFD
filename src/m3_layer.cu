// MS-AFD · M3-mini — one real MoE layer through the fabric's FFN, checked vs HF.
//
// Loads what tools/dump_reference.py dumped for one Qwen3-30B-A3B MoE layer:
//   real expert weights, the layer's input activations, the router's top-k
//   assignments, and HF's reference output. Runs each active expert on its
//   assigned tokens (capacity-padded so the shape stays fixed — the M2 fabric's
//   requirement), combines per token with the router weights, and compares the
//   result to HF's output (relative L2).
//
// This proves the fabric handles REAL weights under REAL dynamic routing
// correctly. (Single process here — determinism-through-the-fabric is a separate
// step; this is the correctness gate.)
//
// Usage: m3_layer <ref_dir>   (default results/m3_ref)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "check.h"
#include "arena.h"
#include "ffn.h"

using namespace msafd;

template <typename T>
static std::vector<T> read_bin(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<T> v(count);
    f.read(reinterpret_cast<char*>(v.data()), count * sizeof(T));
    if ((size_t)f.gcount() != count * sizeof(T)) {
        std::fprintf(stderr, "short read %s: got %zd want %zu\n", path.c_str(),
                     (ssize_t)f.gcount(), count * sizeof(T));
        std::exit(1);
    }
    return v;
}

static float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "results/m3_ref";

    int T, D, F, nexp, topk;
    {
        std::ifstream m(dir + "/meta.txt");
        if (!m || !(m >> T >> D >> F >> nexp >> topk)) {
            std::fprintf(stderr, "cannot read %s/meta.txt\n", dir.c_str());
            return 1;
        }
    }
    std::printf("M3 layer: T=%d d_model=%d d_ff=%d experts=%d top_k=%d\n",
                T, D, F, nexp, topk);

    // Load activations, routing, reference.
    auto hidden = read_bin<__nv_bfloat16>(dir + "/hidden_in.bin", (size_t)T * D);
    auto ref    = read_bin<__nv_bfloat16>(dir + "/moe_out.bin",   (size_t)T * D);
    auto idx    = read_bin<int32_t>(dir + "/topk_idx.bin", (size_t)T * topk);
    auto wgt    = read_bin<float>(dir + "/topk_w.bin",     (size_t)T * topk);

    // Group tokens by expert; capacity C = max per-expert load.
    std::vector<std::vector<int>> tokens_of(nexp);   // expert -> token ids
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < topk; ++k)
            tokens_of[idx[(size_t)t * topk + k]].push_back(t);
    int C = 0;
    for (auto& v : tokens_of) C = std::max(C, (int)v.size());
    if (C == 0) { std::fprintf(stderr, "no routed tokens?\n"); return 1; }
    std::printf("capacity C = %d (max tokens on any expert)\n", C);

    // Fabric FFN sized to the capacity tile.
    FfnConfig cfg;
    cfg.d_model = D; cfg.d_intermediate = F; cfg.tokens = C;
    Arena arena;
    Ffn ffn(cfg, arena);
    const size_t tile = (size_t)C * D;
    void* d_in  = arena.alloc(tile * sizeof(__nv_bfloat16), "m3_in");
    void* d_out = arena.alloc(tile * sizeof(__nv_bfloat16), "m3_out");
    const size_t ws = 4ull << 20;
    void* d_ws = arena.alloc(ws, "cublas_ws");
    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetWorkspace(handle, d_ws, ws));
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<double> out((size_t)T * D, 0.0);   // combined output (fp32)
    std::vector<__nv_bfloat16> h_in(tile), h_out(tile);
    const size_t wgu = (size_t)D * F, wd = (size_t)F * D;
    int active = 0;

    for (int e = 0; e < nexp; ++e) {
        const auto& toks = tokens_of[e];
        if (toks.empty()) continue;
        ++active;

        // Load this expert's real weights.
        auto g = read_bin<__nv_bfloat16>(dir + "/experts/e" + std::to_string(e) + "_gate.bin", wgu);
        auto u = read_bin<__nv_bfloat16>(dir + "/experts/e" + std::to_string(e) + "_up.bin",   wgu);
        auto d = read_bin<__nv_bfloat16>(dir + "/experts/e" + std::to_string(e) + "_down.bin", wd);
        ffn.load_weights_bf16(g.data(), u.data(), d.data());

        // Build the capacity tile: real tokens in slots [0..load), rest zero-pad.
        std::fill(h_in.begin(), h_in.end(), __float2bfloat16(0.0f));
        for (size_t s = 0; s < toks.size(); ++s)
            for (int j = 0; j < D; ++j)
                h_in[s * D + j] = hidden[(size_t)toks[s] * D + j];
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), tile * sizeof(__nv_bfloat16),
                              cudaMemcpyHostToDevice));

        ffn.forward(d_in, d_out, handle, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, tile * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));

        // Scatter-combine into the per-token output, weighted by the router.
        for (size_t s = 0; s < toks.size(); ++s) {
            int t = toks[s];
            float w = 0.0f;
            for (int k = 0; k < topk; ++k)
                if (idx[(size_t)t * topk + k] == e) { w = wgt[(size_t)t * topk + k]; break; }
            for (int j = 0; j < D; ++j)
                out[(size_t)t * D + j] += (double)w * bf2f(h_out[s * D + j]);
        }
    }
    std::printf("ran %d active experts (of %d)\n", active, nexp);

    // Correctness: relative L2 vs HF reference.
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < (size_t)T * D; ++i) {
        double r = bf2f(ref[i]);
        double diff = out[i] - r;
        num += diff * diff; den += r * r;
    }
    double rel_l2 = std::sqrt(num) / (std::sqrt(den) + 1e-12);
    std::printf("relative L2 vs HF moe_out = %.5f\n", rel_l2);
    if (rel_l2 > 0.05) {
        std::fprintf(stderr, "FAIL: rel L2 > 0.05 — check gate/up split order or "
                             "weight orientation in dump_reference.py\n");
        return 1;
    }
    std::printf("M3-mini correctness gate PASSED — real weights + real routing "
                "reproduced through the fabric FFN\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
