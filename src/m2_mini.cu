// MS-AFD · M2-mini — one cross-GPU FFN hop (the smallest real fabric).
//
// Rank 0 (A-side stub, GPU 0): each beat, send a fixed token tile to the F-side
//   and receive the FFN result back.
// Rank 1 (F-side expert, GPU 1): each beat, receive the tile, run the FFN
//   (reusing the M0/M1 Ffn), send the result back.
//
// The F-side hot path — ncclRecv -> FFN GEMMs (cuBLAS) -> ncclSend — is captured
// into ONE CUDA Graph, combining S1's graph-captured NCCL with M0's graph-
// captured cuBLAS. The A-side round-trip (send -> recv) is captured too.
//
// Headline metric: the A-side per-beat p99/p50 — the determinism of the full
// routed pipeline, INCLUDING both NCCL hops. This is the M2 question at minimal
// scale. One rank per GPU (NCCL rejects two ranks on one GPU; see spike S1).
//
// Usage: m2_mini <rank:0|1> <uniqueid_file> [beats] [out.csv]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <nccl.h>

#include "check.h"     // CUDA_CHECK, CUBLAS_CHECK
#include "arena.h"
#include "ffn.h"

using namespace msafd;

#define NCCL_CHECK(call)                                                       \
    do {                                                                       \
        ncclResult_t r__ = (call);                                            \
        if (r__ != ncclSuccess) {                                             \
            std::fprintf(stderr, "NCCL error %d (%s) at %s:%d\n", (int)r__,   \
                         ncclGetErrorString(r__), __FILE__, __LINE__);        \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Share the NCCL unique id via the filesystem (same handshake as spike S1).
static void exchange_id(int rank, const char* idfile, ncclUniqueId* id) {
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(id));
        std::string tmp = std::string(idfile) + ".tmp";
        { std::ofstream f(tmp, std::ios::binary);
          f.write(reinterpret_cast<char*>(id), sizeof(*id)); }
        std::rename(tmp.c_str(), idfile);  // atomic publish
    } else {
        for (int tries = 0;; ++tries) {
            std::ifstream f(idfile, std::ios::binary | std::ios::ate);
            if (f && f.tellg() == (std::streamoff)sizeof(*id)) {
                f.seekg(0);
                f.read(reinterpret_cast<char*>(id), sizeof(*id));
                return;
            }
            if (tries > 1000) { std::fprintf(stderr, "rank1: id timeout\n"); std::exit(1); }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
}

static void write_csv(const char* path, const std::vector<float>& ms) {
    FILE* f = std::fopen(path, "w");
    if (!f) { std::perror(path); std::exit(1); }
    std::fprintf(f, "iter,ms\n");
    for (size_t i = 0; i < ms.size(); ++i) std::fprintf(f, "%zu,%.6f\n", i, ms[i]);
    std::fclose(f);
}

static void report(const char* tag, const char* csv, std::vector<float> ms) {
    write_csv(csv, ms);
    std::sort(ms.begin(), ms.end());
    auto pct = [&](double p) { return ms[(size_t)(p * (ms.size() - 1))]; };
    std::printf("[%s] wrote %s | p50=%.4f ms  p99=%.4f ms  p99/p50=%.3f\n", tag,
                csv, pct(0.50), pct(0.99), pct(0.99) / pct(0.50));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <rank:0|1> <uniqueid_file> [beats] [out.csv]\n",
                     argv[0]);
        return 2;
    }
    const int rank = std::atoi(argv[1]);
    const char* idfile = argv[2];
    const int beats = (argc > 3) ? std::atoi(argv[3]) : 10000;
    const char* csv = (argc > 4) ? argv[4]
                                 : (rank == 0 ? "m2_aside.csv" : "m2_fside.csv");
    const int nranks = 2, peer = 1 - rank, warmup = 200;

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < nranks) {
        std::fprintf(stderr, "M2-mini needs >=2 GPUs (found %d)\n", ndev);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(rank));  // 1 rank per GPU

    ncclUniqueId id;
    exchange_id(rank, idfile, &id);
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    FfnConfig cfg;
    const int T = cfg.tokens, D = cfg.d_model;
    const size_t elems = (size_t)T * D;
    const size_t bytes = elems * sizeof(__nv_bfloat16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times(beats);

    if (rank == 1) {
        // ---------------- F-side: recv -> FFN -> send ------------------------
        std::printf("[F-side] GPU %d: building expert (Qwen3-30B-A3B)\n", rank);
        Arena arena;
        Ffn ffn(cfg, arena);
        void* d_in  = arena.alloc(bytes, "recv_in");
        void* d_out = arena.alloc(bytes, "send_out");
        const size_t ws_bytes = 4ull << 20;
        void* d_ws  = arena.alloc(ws_bytes, "cublas_ws");
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetWorkspace(handle, d_ws, ws_bytes));

        // Pre-warm: eager recv -> forward -> send (warms cuBLAS + NCCL channels
        // so nothing initializes/allocates during capture).
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclRecv(d_in, elems, ncclBfloat16, peer, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        ffn.forward(d_in, d_out, handle, stream);
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(d_out, elems, ncclBfloat16, peer, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Capture the whole hot path into one graph.
        cudaGraph_t graph; cudaGraphExec_t exec;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclRecv(d_in, elems, ncclBfloat16, peer, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        ffn.forward(d_in, d_out, handle, stream);
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(d_out, elems, ncclBfloat16, peer, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
        std::printf("[F-side] hot path captured (recv -> FFN -> send)\n");

        for (int i = 0; i < warmup; ++i) CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < beats; ++i) {
            CUDA_CHECK(cudaEventRecord(start, stream));
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
        }
        report("F-side", csv, times);

        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUBLAS_CHECK(cublasDestroy(handle));
    } else {
        // ---------------- A-side stub: send tokens -> recv result ------------
        std::printf("[A-side] GPU %d: token-tile driver\n", rank);
        Arena arena;
        void* d_tokens = arena.alloc(bytes, "tokens");
        void* d_result = arena.alloc(bytes, "result");

        std::vector<float> h(elems);
        std::mt19937 rng(99);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : h) v = dist(rng);
        std::vector<__nv_bfloat16> hbf(elems);
        for (size_t i = 0; i < elems; ++i) hbf[i] = __float2bfloat16(h[i]);
        CUDA_CHECK(cudaMemcpy(d_tokens, hbf.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_result, 0, bytes));

        // Pre-warm round trip.
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(d_tokens, elems, ncclBfloat16, peer, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclRecv(d_result, elems, ncclBfloat16, peer, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Sanity: result came back non-zero. Snapshot it to check determinism.
        std::vector<__nv_bfloat16> first(elems);
        CUDA_CHECK(cudaMemcpy(first.data(), d_result, bytes, cudaMemcpyDeviceToHost));
        int nonzero = 0;
        for (size_t i = 0; i < elems; ++i)
            if (__bfloat162float(first[i]) != 0.0f) ++nonzero;
        std::printf("[A-side] round-trip returned %d/%zu non-zero elements\n",
                    nonzero, elems);
        if (nonzero == 0) {
            std::fprintf(stderr, "[A-side] FAIL: no data returned from F-side\n");
            return 1;
        }

        // Capture the round trip.
        cudaGraph_t graph; cudaGraphExec_t exec;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(d_tokens, elems, ncclBfloat16, peer, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclRecv(d_result, elems, ncclBfloat16, peer, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
        std::printf("[A-side] round trip captured (send -> recv)\n");

        for (int i = 0; i < warmup; ++i) CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < beats; ++i) {
            CUDA_CHECK(cudaEventRecord(start, stream));
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
        }

        // Determinism check: same fixed input every beat -> identical output.
        std::vector<__nv_bfloat16> last(elems);
        CUDA_CHECK(cudaMemcpy(last.data(), d_result, bytes, cudaMemcpyDeviceToHost));
        bool stable = true;
        for (size_t i = 0; i < elems; ++i)
            if (__bfloat162float(last[i]) != __bfloat162float(first[i])) { stable = false; break; }
        std::printf("[A-side] routed output %s across beats\n",
                    stable ? "STABLE (bit-identical)" : "CHANGED (nondeterministic!)");
        report("A-side (round-trip beat)", csv, times);

        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));
    if (rank == 0) std::remove(idfile);
    return 0;
}
