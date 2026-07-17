// MS-AFD · M0 — Single slice.
// One MPS process: static arena -> FFN captured into a CUDA Graph -> persistent
// loop, timing each iteration. Emits latency.csv for bench/latency.py.
//
// Run under MPS (see scripts/start_mps.sh); CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
// caps this process to a fraction of the GPU's SMs (Step 1 — external).
//
// Usage: ./slice [iters] [tokens] [out.csv]
//   iters   = timed graph launches (default 5000)
//   tokens  = token batch per iteration (default 256)
//   out.csv = per-iteration timing output path (default latency.csv)
// Env: MSAFD_ARENA_MB overrides the arena size in MiB (default 1024). Used in M1
//      to pack many slices into one GPU's HBM.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "arena.h"
#include "ffn.h"

using namespace msafd;

// Cross-process barrier for M1: every slice drops a ready-file, then spins until
// it sees all N of them, so all slices enter the timed loop simultaneously
// (otherwise staggered CPU-reference phases would skew the contention measured).
// No-op unless MSAFD_BARRIER_DIR is set. MSAFD_BARRIER_N = expected slice count.
static void barrier_wait() {
    const char* dir = std::getenv("MSAFD_BARRIER_DIR");
    if (!dir) return;
    int need = std::getenv("MSAFD_BARRIER_N")
                   ? std::atoi(std::getenv("MSAFD_BARRIER_N"))
                   : 1;
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/ready_%d", dir, (int)getpid());
    if (FILE* rf = std::fopen(path, "w")) {
        std::fputc('1', rf);
        std::fclose(rf);
    }
    for (int spins = 0; spins < 60000; ++spins) {  // ~120 s timeout
        int count = 0;
        if (DIR* d = opendir(dir)) {
            for (dirent* e; (e = readdir(d));)
                if (std::strncmp(e->d_name, "ready_", 6) == 0) ++count;
            closedir(d);
        }
        if (count >= need) return;
        usleep(2000);
    }
    fprintf(stderr, "[barrier] timeout waiting for %d slices\n", need);
}

// CPU reference for one output row (token t0): full SwiGLU FFN in fp32, reading
// the host mirror of the weights. Fills ref[0..d_model).
static void cpu_reference_row(const FfnConfig& cfg, const std::vector<__half>& in,
                              const std::vector<__half>& Wg,
                              const std::vector<__half>& Wu,
                              const std::vector<__half>& Wd, int t0,
                              std::vector<float>& ref) {
    const int dm = cfg.d_model, di = cfg.d_intermediate;
    std::vector<float> h(di);
    for (int i = 0; i < di; ++i) {
        float g = 0.0f, u = 0.0f;
        for (int k = 0; k < dm; ++k) {
            float x = __half2float(in[(std::size_t)t0 * dm + k]);
            g += x * __half2float(Wg[(std::size_t)i * dm + k]);
            u += x * __half2float(Wu[(std::size_t)i * dm + k]);
        }
        float s = g / (1.0f + std::exp(-g));  // SiLU
        h[i]    = s * u;
    }
    ref.assign(dm, 0.0f);
    for (int j = 0; j < dm; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < di; ++i)
            acc += h[i] * __half2float(Wd[(std::size_t)j * di + i]);
        ref[j] = acc;
    }
}

int main(int argc, char** argv) {
    const int   iters   = argc > 1 ? std::atoi(argv[1]) : 5000;
    const int   tokens  = argc > 2 ? std::atoi(argv[2]) : 256;
    const char* out_csv = argc > 3 ? argv[3] : "latency.csv";
    const int   warmup  = 200;

    // --- Step 2: static arena (1 GB default; MSAFD_ARENA_MB to pack slices) ---
    std::size_t arena_bytes = kArenaBytes;
    if (const char* mb = std::getenv("MSAFD_ARENA_MB"))
        arena_bytes = (std::size_t)std::atoll(mb) << 20;
    Arena arena(arena_bytes);

    FfnConfig cfg;
    cfg.tokens = tokens;
    Ffn ffn(cfg, arena);
    const int T = cfg.tokens, dm = cfg.d_model;

    __half* input  = (__half*)arena.alloc(sizeof(__half) * T * dm, "act_input");
    __half* output = (__half*)arena.alloc(sizeof(__half) * T * dm, "act_output");

    // Deterministic input fill (host + device).
    std::vector<__half> h_in((std::size_t)T * dm);
    for (std::size_t i = 0; i < h_in.size(); ++i) {
        unsigned x = (unsigned)(i * 2246822519u + 11u);
        float    r = ((x >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
        h_in[i]    = __float2half(r * 0.05f);
    }
    MSAFD_CUDA_CHECK(cudaMemcpy(input, h_in.data(),
                                h_in.size() * sizeof(__half),
                                cudaMemcpyHostToDevice));

    printf("[slice] arena used %zu / %zu bytes; tokens=%d d_model=%d d_int=%d\n",
           arena.used(), arena.capacity(), T, dm, cfg.d_intermediate);

    cudaStream_t stream;
    MSAFD_CUDA_CHECK(cudaStreamCreate(&stream));

    // --- Step 3: eager forward + correctness ---------------------------------
    ffn.forward(input, output, stream);
    MSAFD_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__half> eager((std::size_t)T * dm);
    MSAFD_CUDA_CHECK(cudaMemcpy(eager.data(), output,
                                eager.size() * sizeof(__half),
                                cudaMemcpyDeviceToHost));

    // NaN / inf scan of the whole output.
    for (std::size_t i = 0; i < eager.size(); ++i) {
        float v = __half2float(eager[i]);
        if (std::isnan(v) || std::isinf(v)) {
            fprintf(stderr, "[check] non-finite output at %zu\n", i);
            std::abort();
        }
    }

    // CPU spot-check: token 0, first 8 output channels.
    std::vector<float> ref;
    cpu_reference_row(cfg, h_in, ffn.host_wg(), ffn.host_wu(), ffn.host_wd(), 0,
                      ref);
    double max_rel = 0.0;
    for (int j = 0; j < 8; ++j) {
        float got = __half2float(eager[j]);
        float exp = ref[j];
        float rel = std::fabs(got - exp) / (1e-3f + std::fabs(exp));
        max_rel   = std::max(max_rel, (double)rel);
        printf("[check] out[0][%d] gpu=% .6f cpu=% .6f rel=%.4f\n", j, got, exp,
               rel);
    }
    if (max_rel > 0.05) {
        fprintf(stderr, "[check] FAILED: max rel err %.4f > 0.05\n", max_rel);
        std::abort();
    }
    printf("[check] eager vs CPU reference OK (max rel %.4f)\n", max_rel);

    // --- Step 4: capture the FFN into a CUDA Graph ---------------------------
    cudaGraph_t     graph;
    cudaGraphExec_t exec;
    MSAFD_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ffn.forward(input, output, stream);
    MSAFD_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    MSAFD_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

    MSAFD_CUDA_CHECK(cudaGraphLaunch(exec, stream));
    MSAFD_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__half> graphed((std::size_t)T * dm);
    MSAFD_CUDA_CHECK(cudaMemcpy(graphed.data(), output,
                                graphed.size() * sizeof(__half),
                                cudaMemcpyDeviceToHost));
    double max_abs = 0.0;
    for (std::size_t i = 0; i < graphed.size(); ++i)
        max_abs = std::max(max_abs, (double)std::fabs(__half2float(graphed[i]) -
                                                      __half2float(eager[i])));
    printf("[check] graph vs eager: max abs diff %.6g %s\n", max_abs,
           max_abs == 0.0 ? "(bit-exact)" : "(within tolerance)");

    // --- Step 5 + 6: persistent loop, time every iteration -------------------
    cudaEvent_t start, stop;
    MSAFD_CUDA_CHECK(cudaEventCreate(&start));
    MSAFD_CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        MSAFD_CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }
    MSAFD_CUDA_CHECK(cudaStreamSynchronize(stream));

    // M1: wait until all concurrent slices are warmed up, so the timed loops
    // overlap and we measure real steady-state contention (no-op in M0).
    barrier_wait();

    std::vector<float> ms(iters);
    for (int i = 0; i < iters; ++i) {
        MSAFD_CUDA_CHECK(cudaEventRecord(start, stream));
        MSAFD_CUDA_CHECK(cudaGraphLaunch(exec, stream));
        MSAFD_CUDA_CHECK(cudaEventRecord(stop, stream));
        MSAFD_CUDA_CHECK(cudaEventSynchronize(stop));
        MSAFD_CUDA_CHECK(cudaEventElapsedTime(&ms[i], start, stop));
    }

    FILE* f = std::fopen(out_csv, "w");
    if (!f) {
        perror("[slice] fopen out_csv");
        return 1;
    }
    std::fprintf(f, "iter,ms\n");
    for (int i = 0; i < iters; ++i) std::fprintf(f, "%d,%.6f\n", i, ms[i]);
    std::fclose(f);

    // Quick inline summary (bench/latency.py does the full report).
    std::vector<float> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) {
        double idx = p / 100.0 * (sorted.size() - 1);
        std::size_t lo = (std::size_t)idx;
        double frac = idx - lo;
        if (lo + 1 >= sorted.size()) return (double)sorted.back();
        return sorted[lo] * (1 - frac) + sorted[lo + 1] * frac;
    };
    double p50 = pct(50), p99 = pct(99);
    printf("[loop] %d iters | p50=%.4f ms p99=%.4f ms p99/p50=%.3f\n", iters,
           p50, p99, p99 / p50);
    printf("[slice] wrote %s\n", out_csv);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return 0;
}
