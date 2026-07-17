// MS-AFD · M0/M1 — One slice process.
// One MPS process: static arena -> FFN captured into a CUDA Graph -> persistent
// loop, timing each iteration.
//
// Run under MPS (see scripts/start_mps.sh); CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
// caps this process to a fraction of the GPU's SMs (Step 1 — external to this
// file). We do NOT care about latency; the headline number is the shape of the
// per-iteration distribution: p50, p99, and the p99/p50 ratio.
//
// Sequence: arena -> build FFN -> eager forward -> fp32 correctness gate ->
// capture into a CUDA Graph -> warm up -> timed persistent loop -> CSV.
//
// Usage: slice [iters] [out.csv]
//   iters     iterations of the timed loop (default 10000)
//   out.csv   per-iteration timings (default latency.csv). M1 gives each fleet
//             process its own path so they don't clobber each other.
// Env: MSAFD_CHECK=0 skips the fp32 correctness gate. M0 already verified
//   correctness; in an M1 fleet, 48 processes each recomputing the single-
//   threaded CPU reference at once would thrash the host and skew startup.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "check.h"
#include "arena.h"
#include "ffn.h"

using namespace msafd;

int main(int argc, char** argv) {
    const int iters = (argc > 1) ? std::atoi(argv[1]) : 10000;
    const char* csv_path = (argc > 2) ? argv[2] : "latency.csv";
    const char* check_env = std::getenv("MSAFD_CHECK");
    const bool do_check = !(check_env && std::atoi(check_env) == 0);
    const int warmup = 200;

    // --- device / MPS context ------------------------------------------------
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const char* mps_pct = std::getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
    std::printf("MS-AFD slice [%s] on %s (%d SMs, sm_%d%d) | MPS thread%% = %s\n",
                csv_path, prop.name, prop.multiProcessorCount, prop.major,
                prop.minor, mps_pct ? mps_pct : "(unset)");

    // --- Step 2: single static arena -----------------------------------------
    Arena arena;

    // --- Step 3: build one FFN (weights uploaded to HBM) ---------------------
    FfnConfig cfg;
    Ffn ffn(cfg, arena);
    const int T = cfg.tokens, D = cfg.d_model;
    const size_t act_elems = (size_t)T * D;
    const size_t act_bytes = act_elems * sizeof(__nv_bfloat16);

    // Fixed input/output regions in the arena.
    void* d_in  = arena.alloc(act_bytes, "input");
    void* d_out = arena.alloc(act_bytes, "output");

    // Fixed cuBLAS workspace in the arena, so no cudaMalloc happens during graph
    // capture. (Eager forward below also pre-warms cuBLAS.)
    const size_t ws_bytes = 4ull << 20;
    void* d_ws = arena.alloc(ws_bytes, "cublas_ws");

    std::printf("arena: %zu / %zu bytes used (%.1f MB)\n", arena.used(),
                arena.capacity(), arena.used() / (1024.0 * 1024.0));

    // --- host input: fp32 (for reference) + bf16 (for device) ----------------
    std::vector<float> h_in(act_elems);
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_in) v = dist(rng);
    std::vector<__nv_bfloat16> h_in_bf(act_elems);
    for (size_t i = 0; i < act_elems; ++i) h_in_bf[i] = __float2bfloat16(h_in[i]);
    CUDA_CHECK(cudaMemcpy(d_in, h_in_bf.data(), act_bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetWorkspace(handle, d_ws, ws_bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // --- Step 3: eager forward (also pre-warms cuBLAS before capture) --------
    ffn.forward(d_in, d_out, handle, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // fp32 correctness gate (skipped in fleet mode via MSAFD_CHECK=0).
    if (do_check) {
        std::vector<__nv_bfloat16> h_out_bf(act_elems);
        CUDA_CHECK(cudaMemcpy(h_out_bf.data(), d_out, act_bytes,
                              cudaMemcpyDeviceToHost));
        std::vector<float> h_out(act_elems);
        for (size_t i = 0; i < act_elems; ++i)
            h_out[i] = __bfloat162float(h_out_bf[i]);

        std::vector<float> ref(act_elems);
        ffn.reference(h_in.data(), ref.data());

        // Relative L2 norm ||device - ref|| / ||ref||. Robust to near-zero
        // outputs (a per-element ratio would blow up on pure bf16 rounding).
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < act_elems; ++i) {
            double d = (double)h_out[i] - (double)ref[i];
            num += d * d;
            den += (double)ref[i] * (double)ref[i];
        }
        double rel_l2 = std::sqrt(num) / (std::sqrt(den) + 1e-12);
        std::printf("correctness: relative L2 error vs fp32 reference = %.5f\n",
                    rel_l2);
        if (rel_l2 > 0.03) {
            std::fprintf(stderr, "FAIL: correctness gate exceeded (rel L2 > 0.03)\n");
            return 1;
        }
        std::printf("correctness gate PASSED\n");
    } else {
        std::printf("correctness gate SKIPPED (MSAFD_CHECK=0)\n");
    }

    // --- Step 4: capture the FFN into a CUDA Graph ---------------------------
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ffn.forward(d_in, d_out, handle, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    std::printf("CUDA Graph captured + instantiated\n");

    // --- Step 5: warm up, then timed persistent loop -------------------------
    for (int i = 0; i < warmup; ++i) CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> ms(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], start, stop));
    }

    // --- Step 6: dump per-iteration timings for bench/latency.py -------------
    FILE* f = std::fopen(csv_path, "w");
    if (!f) { std::perror(csv_path); return 1; }
    std::fprintf(f, "iter,ms\n");
    for (int i = 0; i < iters; ++i) std::fprintf(f, "%d,%.6f\n", i, ms[i]);
    std::fclose(f);
    std::printf("wrote %s (%d iterations)\n", csv_path, iters);

    // Quick inline p50/p99 so the slice is useful without the Python step.
    std::vector<float> sorted(ms);
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) { return sorted[(size_t)(p * (iters - 1))]; };
    const float p50 = pct(0.50), p99 = pct(0.99);
    std::printf("[%s] p50=%.4f ms  p99=%.4f ms  p99/p50=%.3f\n", csv_path, p50,
                p99, p99 / p50);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
