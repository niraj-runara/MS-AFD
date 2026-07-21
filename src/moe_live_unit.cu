// MS-AFD · M4 — a live MoE micro-unit for one real transformer layer.
//
// One MPS-capped process on an F-side GPU. Loads layer L's real Qwen3 expert
// weights (fused HF layout), opens the coordinator's IPC input/output tiles
// (which live on HF's GPU — cross-GPU over NVLink), then serves: whenever the
// control block names this layer, it P2P-copies the hidden tile local, runs the
// capacity-padded MoE FFN with that dispatch's routing, and P2P-copies the result
// back. Times every T=1 (decode) dispatch -> per-unit determinism during real
// generation.
//
// Usage: moe_live_unit <layer> <n_layers> <D> <F> <n_exp> <top_k> <T_max>
//                      <ctrl> <hdir> <weights_dir> <device> [csv]

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

// C[M,N] = A[M,K] @ Bhf[N,K]^T  (Bhf = HF fused [out=N, in=K] weight block).
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
__global__ void waxpy_k(bf16* out, const bf16* src, float w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(out[i]) + w * __bfloat162float(src[i]));
}

static bf16* load_weight(const std::string& path, size_t elems) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "[unit] cannot open %s\n", path.c_str()); std::abort(); }
    std::vector<char> host(elems * sizeof(bf16));
    size_t got = std::fread(host.data(), 1, host.size(), f);
    std::fclose(f);
    if (got != host.size()) { std::fprintf(stderr, "[unit] short read %s\n", path.c_str()); std::abort(); }
    bf16* d = nullptr;
    LIVE_CUDA_CHECK(cudaMalloc(&d, host.size()));
    LIVE_CUDA_CHECK(cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    if (argc < 12) {
        std::fprintf(stderr, "usage: %s layer n_layers D F n_exp top_k T_max ctrl hdir wdir device [csv]\n", argv[0]);
        return 2;
    }
    const int L = std::atoi(argv[1]), n_layers = std::atoi(argv[2]);
    const int D = std::atoi(argv[3]), F = std::atoi(argv[4]);
    const int n_exp = std::atoi(argv[5]), top_k = std::atoi(argv[6]);
    const int T_max = std::atoi(argv[7]);
    const char* ctrl = argv[8]; const char* hdir = argv[9]; const char* wdir = argv[10];
    const int device = std::atoi(argv[11]);
    const char* csv = (argc > 12) ? argv[12] : nullptr;

    LIVE_CUDA_CHECK(cudaSetDevice(device));
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);   // MPS-fleet discipline
    const char* pct = std::getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");

    // This layer's real experts (fused HF layout): gate_up [n_exp,2F,D], down [n_exp,D,F].
    std::string base = std::string(wdir) + "/L" + std::to_string(L);
    bf16* GU = load_weight(base + "_gate_up.bin", (size_t)n_exp * 2 * F * D);
    bf16* DN = load_weight(base + "_down.bin",    (size_t)n_exp * D * F);

    // Local scratch (this GPU).
    bf16 *lin, *lout, *tin, *g, *u, *hid, *tout;
    LIVE_CUDA_CHECK(cudaMalloc(&lin,  (size_t)T_max * D * sizeof(bf16)));
    LIVE_CUDA_CHECK(cudaMalloc(&lout, (size_t)T_max * D * sizeof(bf16)));
    LIVE_CUDA_CHECK(cudaMalloc(&tin,  (size_t)T_max * D * sizeof(bf16)));
    LIVE_CUDA_CHECK(cudaMalloc(&g,    (size_t)T_max * F * sizeof(bf16)));
    LIVE_CUDA_CHECK(cudaMalloc(&u,    (size_t)T_max * F * sizeof(bf16)));
    LIVE_CUDA_CHECK(cudaMalloc(&hid,  (size_t)T_max * F * sizeof(bf16)));
    LIVE_CUDA_CHECK(cudaMalloc(&tout, (size_t)T_max * D * sizeof(bf16)));

    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
    void* ws; size_t ws_bytes = 32ull << 20;
    LIVE_CUDA_CHECK(cudaMalloc(&ws, ws_bytes));
    CUBLAS_CHECK(cublasSetWorkspace(handle, ws, ws_bytes));
    cudaStream_t stream; LIVE_CUDA_CHECK(cudaStreamCreate(&stream));

    // Open the coordinator's IPC tiles (on HF's GPU — cross-GPU/NVLink).
    cudaIpcMemHandle_t hin, hout;
    std::string pin = std::string(hdir) + "/in.ipc", pout = std::string(hdir) + "/out.ipc";
    for (int t = 0;; ++t) {
        if (live_read_handle(pin.c_str(), &hin, sizeof(hin)) &&
            live_read_handle(pout.c_str(), &hout, sizeof(hout))) break;
        if (t > 15000) { std::fprintf(stderr, "[unit %d] IPC timeout\n", L); return 1; }
        usleep(20000);
    }
    void *g_in, *g_out;
    LIVE_CUDA_CHECK(cudaIpcOpenMemHandle(&g_in, hin, cudaIpcMemLazyEnablePeerAccess));
    LIVE_CUDA_CHECK(cudaIpcOpenMemHandle(&g_out, hout, cudaIpcMemLazyEnablePeerAccess));

    const size_t guStride = (size_t)2 * F * D, dnStride = (size_t)D * F;

    // Run the capacity-padded MoE FFN: lin[T,D] + routing -> lout[T,D].
    auto moe = [&](int T, const int* tki, const float* tkw) {
        std::vector<std::vector<int>> toks(n_exp);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < top_k; ++k) toks[tki[(size_t)t * top_k + k]].push_back(t);
        int C = 1; for (auto& v : toks) if ((int)v.size() > C) C = (int)v.size();
        LIVE_CUDA_CHECK(cudaMemsetAsync(lout, 0, (size_t)T * D * sizeof(bf16), stream));
        const int thr = 256;
        for (int e = 0; e < n_exp; ++e) {
            const int load = (int)toks[e].size();
            if (!load) continue;
            LIVE_CUDA_CHECK(cudaMemsetAsync(tin, 0, (size_t)C * D * sizeof(bf16), stream));
            for (int s = 0; s < load; ++s)
                LIVE_CUDA_CHECK(cudaMemcpyAsync(tin + (size_t)s * D, lin + (size_t)toks[e][s] * D,
                                                (size_t)D * sizeof(bf16), cudaMemcpyDeviceToDevice, stream));
            const bf16* gate_hf = GU + (size_t)e * guStride;
            const bf16* up_hf   = GU + (size_t)e * guStride + (size_t)F * D;
            const bf16* down_hf = DN + (size_t)e * dnStride;
            gemm_linear(handle, tin, gate_hf, g, C, D, F, stream);
            gemm_linear(handle, tin, up_hf,   u, C, D, F, stream);
            int n = C * F, blk = (n + thr - 1) / thr;
            swiglu_k<<<blk, thr, 0, stream>>>(g, u, hid, n);
            gemm_linear(handle, hid, down_hf, tout, C, F, D, stream);
            int dblk = (D + thr - 1) / thr;
            for (int s = 0; s < load; ++s) {
                int t = toks[e][s]; float w = 0.0f;
                for (int k = 0; k < top_k; ++k)
                    if (tki[(size_t)t * top_k + k] == e) { w = tkw[(size_t)t * top_k + k]; break; }
                waxpy_k<<<dblk, thr, 0, stream>>>(lout + (size_t)t * D, tout + (size_t)s * D, w, D);
            }
        }
    };

    cudaEvent_t s0, s1;
    LIVE_CUDA_CHECK(cudaEventCreate(&s0)); LIVE_CUDA_CHECK(cudaEventCreate(&s1));

    LiveControl* c = live_map_control(ctrl, false, n_layers);
    std::fprintf(stderr, "[unit %d] ready on GPU%d (MPS%%=%s)\n", L, device, pct ? pct : "unset");
    c->ready[L] = 1;

    std::vector<int> tki(kLiveMaxTokens * top_k);
    std::vector<float> tkw(kLiveMaxTokens * top_k);
    std::vector<float> ms;
    int last = 0;
    for (;;) {
        while (!(c->active_layer == L && c->epoch != last) && !c->stop) sched_yield();
        if (c->stop) break;
        last = c->epoch;
        int T = c->T;
        for (int i = 0; i < T * top_k; ++i) { tki[i] = c->topk_idx[i]; tkw[i] = c->topk_w[i]; }

        LIVE_CUDA_CHECK(cudaEventRecord(s0, stream));
        LIVE_CUDA_CHECK(cudaMemcpyAsync(lin, g_in, (size_t)T * D * sizeof(bf16),
                                        cudaMemcpyDefault, stream));   // GPU0 -> unit GPU (P2P)
        moe(T, tki.data(), tkw.data());
        LIVE_CUDA_CHECK(cudaMemcpyAsync(g_out, lout, (size_t)T * D * sizeof(bf16),
                                        cudaMemcpyDefault, stream));   // unit GPU -> GPU0 (P2P)
        LIVE_CUDA_CHECK(cudaEventRecord(s1, stream));
        LIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (T == 1) { float t; LIVE_CUDA_CHECK(cudaEventElapsedTime(&t, s0, s1)); ms.push_back(t); }
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
