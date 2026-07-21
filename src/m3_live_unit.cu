// MS-AFD · M4 (G1) — a live FFN micro-unit for one real transformer layer.
//
// One MPS-capped process. Loads layer L's real Llama-3 weights, opens the
// coordinator's IPC input/output buffers, captures its FFN (decode shape T=1)
// into a CUDA graph, then serves: whenever the control block names this layer, it
// runs the FFN on the shared input tile and signals done. Times every T=1 (decode)
// FFN it serves -> per-unit determinism *during real generation*.
//
// Usage: m3_live_unit <layer> <n_layers> <D> <F> <T_max> <ctrl> <hdir>
//                     <weights_dir> <device> [csv]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include "check.h"
#include "live_common.h"

using namespace msafd;
using bf16 = __nv_bfloat16;

// C[M,N] = A[M,K] @ Bhf[N,K]^T  (Bhf = HF Linear [out=N, in=K] weight).
static void gemm_linear(cublasHandle_t h, const bf16* A, const bf16* Bhf, bf16* C,
                        int M, int K, int N, cudaStream_t s) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(h, s));
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

static bf16* load_weight(const std::string& path, size_t elems) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "[unit] cannot open %s\n", path.c_str()); std::abort(); }
    std::vector<char> host(elems * sizeof(bf16));
    size_t got = std::fread(host.data(), 1, host.size(), f);
    std::fclose(f);
    if (got != host.size()) {
        std::fprintf(stderr, "[unit] short read %s: %zu/%zu\n", path.c_str(), got, host.size());
        std::abort();
    }
    bf16* d = nullptr;
    LIVE_CUDA_CHECK(cudaMalloc(&d, host.size()));
    LIVE_CUDA_CHECK(cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    if (argc < 10) {
        std::fprintf(stderr, "usage: %s layer n_layers D F T_max ctrl hdir weights_dir device [csv]\n", argv[0]);
        return 2;
    }
    const int   L        = std::atoi(argv[1]);
    const int   n_layers = std::atoi(argv[2]);
    const int   D        = std::atoi(argv[3]);
    const int   F        = std::atoi(argv[4]);
    const int   T_max    = std::atoi(argv[5]);
    const char* ctrl     = argv[6];
    const char* hdir     = argv[7];
    const char* wdir     = argv[8];
    const int   device   = std::atoi(argv[9]);
    const char* csv      = (argc > 10) ? argv[10] : nullptr;

    LIVE_CUDA_CHECK(cudaSetDevice(device));
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);  // MPS-fleet discipline
    const char* pct = std::getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");

    // Load this layer's real weights (HF nn.Linear layout).
    std::string base = std::string(wdir) + "/L" + std::to_string(L);
    bf16* Wg = load_weight(base + "_gate.bin", (size_t)F * D);
    bf16* Wu = load_weight(base + "_up.bin",   (size_t)F * D);
    bf16* Wd = load_weight(base + "_down.bin", (size_t)D * F);
    bf16 *g, *u, *h;
    LIVE_CUDA_CHECK(cudaMalloc(&g, (size_t)T_max * F * sizeof(bf16)));
    LIVE_CUDA_CHECK(cudaMalloc(&u, (size_t)T_max * F * sizeof(bf16)));
    LIVE_CUDA_CHECK(cudaMalloc(&h, (size_t)T_max * F * sizeof(bf16)));

    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
    void* ws; size_t ws_bytes = 32ull << 20;
    LIVE_CUDA_CHECK(cudaMalloc(&ws, ws_bytes));
    CUBLAS_CHECK(cublasSetWorkspace(handle, ws, ws_bytes));
    cudaStream_t stream; LIVE_CUDA_CHECK(cudaStreamCreate(&stream));

    // Wait for the coordinator's IPC buffers, then open them.
    cudaIpcMemHandle_t hin, hout;
    std::string pin = std::string(hdir) + "/in.ipc", pout = std::string(hdir) + "/out.ipc";
    for (int t = 0;; ++t) {
        if (live_read_handle(pin.c_str(), &hin, sizeof(hin)) &&
            live_read_handle(pout.c_str(), &hout, sizeof(hout))) break;
        if (t > 15000) { std::fprintf(stderr, "[unit %d] IPC timeout\n", L); return 1; }
        usleep(20000);
    }
    void *inbase, *outbase;
    LIVE_CUDA_CHECK(cudaIpcOpenMemHandle(&inbase, hin, cudaIpcMemLazyEnablePeerAccess));
    LIVE_CUDA_CHECK(cudaIpcOpenMemHandle(&outbase, hout, cudaIpcMemLazyEnablePeerAccess));
    bf16* in  = static_cast<bf16*>(inbase);
    bf16* out = static_cast<bf16*>(outbase);

    auto forward = [&](int T) {
        gemm_linear(handle, in, Wg, g, T, D, F, stream);
        gemm_linear(handle, in, Wu, u, T, D, F, stream);
        int n = T * F, thr = 256, blk = (n + thr - 1) / thr;
        swiglu_k<<<blk, thr, 0, stream>>>(g, u, h, n);
        gemm_linear(handle, h, Wd, out, T, F, D, stream);
    };

    // Capture the decode-shape (T=1) FFN into a replayable graph.
    forward(1);
    LIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph; cudaGraphExec_t exec;
    LIVE_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    forward(1);
    LIVE_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    LIVE_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

    cudaEvent_t s0, s1;
    LIVE_CUDA_CHECK(cudaEventCreate(&s0));
    LIVE_CUDA_CHECK(cudaEventCreate(&s1));

    LiveControl* c = live_map_control(ctrl, false, n_layers);
    std::fprintf(stderr, "[unit %d] ready on GPU%d (MPS%%=%s)\n", L, device, pct ? pct : "unset");
    c->ready[L] = 1;

    std::vector<float> ms;
    int last = 0;
    for (;;) {
        while (!(c->active_layer == L && c->epoch != last) && !c->stop) sched_yield();
        if (c->stop) break;
        last = c->epoch;
        int T = c->T;
        if (T == 1) {
            LIVE_CUDA_CHECK(cudaEventRecord(s0, stream));
            LIVE_CUDA_CHECK(cudaGraphLaunch(exec, stream));
            LIVE_CUDA_CHECK(cudaEventRecord(s1, stream));
            LIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
            float t; LIVE_CUDA_CHECK(cudaEventElapsedTime(&t, s0, s1));
            ms.push_back(t);
        } else {
            forward(T);
            LIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        c->done_epoch = last;
    }

    if (csv && !ms.empty()) {
        FILE* f = std::fopen(csv, "w");
        std::fprintf(f, "iter,ms\n");
        for (size_t i = 0; i < ms.size(); ++i) std::fprintf(f, "%zu,%.6f\n", i, ms[i]);
        std::fclose(f);
        std::sort(ms.begin(), ms.end());
        auto q = [&](double p){ return ms[(size_t)(p*(ms.size()-1))]; };
        std::fprintf(stderr, "[unit %d] decode beats=%zu p50=%.4f p99=%.4f p99/p50=%.3f -> %s\n",
                     L, ms.size(), q(0.50), q(0.99), q(0.99)/q(0.50), csv);
    }
    return 0;
}
