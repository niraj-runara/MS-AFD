// MS-AFD · Spike S1 — graph-captured NCCL (retires risk R1).
//
// The whole M1/M2 story depends on the FFN living inside a replayable CUDA
// Graph. M2 also needs NCCL send/recv *inside* that graph. R1 asks: is that
// even capture-safe? S1 answers it as cheaply as possible.
//
// Two processes on ONE GPU under MPS, each an NCCL rank. We capture a compute
// kernel + ncclSend + ncclRecv into a SINGLE CUDA Graph and replay it N times.
// PASS = the graph instantiates, replays, and the exchanged data is correct.
// If capture fails, M2 must issue NCCL outside the graph (the plan's fallback).
//
// Usage: s1_graph_nccl <rank:0|1> <uniqueid_file>
//   rank 0 generates the NCCL unique id and writes it (atomically) to the file;
//   rank 1 waits for the file and reads it. Launch both (see scripts/run_s1.sh).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e__ = (call);                                             \
        if (e__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                         cudaGetErrorName(e__), __FILE__, __LINE__,           \
                         cudaGetErrorString(e__));                            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define NCCL_CHECK(call)                                                       \
    do {                                                                       \
        ncclResult_t r__ = (call);                                            \
        if (r__ != ncclSuccess) {                                             \
            std::fprintf(stderr, "NCCL error %d (%s) at %s:%d\n", (int)r__,   \
                         ncclGetErrorString(r__), __FILE__, __LINE__);        \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Trivial compute to prove a kernel and NCCL comms co-capture into one graph.
__global__ void bump_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <rank:0|1> <uniqueid_file>\n", argv[0]);
        return 2;
    }
    const int rank = std::atoi(argv[1]);
    const char* idfile = argv[2];
    const int nranks = 2;
    const int peer = 1 - rank;
    const int N = 1024;
    const int replays = 10;

    // --- share the NCCL unique id via the filesystem -------------------------
    ncclUniqueId id;
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&id));
        std::string tmp = std::string(idfile) + ".tmp";
        { std::ofstream f(tmp, std::ios::binary);
          f.write(reinterpret_cast<char*>(&id), sizeof(id)); }
        std::rename(tmp.c_str(), idfile);  // atomic publish
    } else {
        for (int tries = 0;; ++tries) {
            std::ifstream f(idfile, std::ios::binary | std::ios::ate);
            if (f && f.tellg() == (std::streamoff)sizeof(id)) {
                f.seekg(0);
                f.read(reinterpret_cast<char*>(&id), sizeof(id));
                break;
            }
            if (tries > 1000) {
                std::fprintf(stderr, "rank1: timed out waiting for %s\n", idfile);
                return 1;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    // --- pick this rank's device --------------------------------------------
    // NCCL rejects two ranks sharing one physical GPU ("Duplicate GPU detected",
    // ncclInvalidUsage) even under MPS. So when >1 GPU is visible we bind rank r
    // to device r (1 rank/GPU) — which is also how M2 routes (cross-GPU). With a
    // single GPU both ranks land on device 0 and NCCL will (expectedly) refuse.
    // Optional 3rd arg overrides the device explicitly.
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    const int dev = (argc > 3) ? std::atoi(argv[3]) : (ndev >= nranks ? rank : 0);
    CUDA_CHECK(cudaSetDevice(dev));
    std::printf("[rank %d] using device %d of %d visible\n", rank, dev, ndev);
    if (ndev < nranks)
        std::fprintf(stderr,
                     "[rank %d] WARNING: only %d GPU(s) visible; 2 ranks on one "
                     "GPU — NCCL will likely reject this. Use a >=2-GPU box.\n",
                     rank, ndev);

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));
    std::printf("[rank %d] comm initialized (2 ranks, device %d)\n", rank, dev);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float *sendbuf, *recvbuf;
    CUDA_CHECK(cudaMalloc(&sendbuf, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recvbuf, N * sizeof(float)));
    std::vector<float> h(N, (float)(rank + 1));  // rank0 -> 1.0, rank1 -> 2.0
    CUDA_CHECK(cudaMemcpy(sendbuf, h.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(recvbuf, 0, N * sizeof(float)));

    // Pre-warm NCCL once outside capture (sets up channels), mirroring the
    // cuBLAS pre-warm in slice.cu so nothing initializes during capture.
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(sendbuf, N, ncclFloat, peer, comm, stream));
    NCCL_CHECK(ncclRecv(recvbuf, N, ncclFloat, peer, comm, stream));
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // --- the actual test: capture kernel + send/recv into ONE graph ----------
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    bump_kernel<<<(N + 255) / 256, 256, 0, stream>>>(sendbuf, N);
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(sendbuf, N, ncclFloat, peer, comm, stream));
    NCCL_CHECK(ncclRecv(recvbuf, N, ncclFloat, peer, comm, stream));
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    std::printf("[rank %d] graph captured + instantiated\n", rank);

    for (int i = 0; i < replays; ++i)
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // --- validate exchanged data --------------------------------------------
    // Each replay bumps sendbuf by 1 then sends it, so after `replays` steps the
    // peer's final sent value is (peer+1)+replays.
    std::vector<float> got(N);
    CUDA_CHECK(cudaMemcpy(got.data(), recvbuf, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    const float expected = (float)(peer + 1) + (float)replays;
    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (got[i] != expected) ++bad;

    int rc = 0;
    if (bad == 0) {
        std::printf("[rank %d] S1 PASS: graph-captured NCCL replayed; "
                    "recv == %.1f (expected from peer %d)\n",
                    rank, expected, peer);
    } else {
        std::fprintf(stderr,
                     "[rank %d] S1 FAIL: %d/%d elements wrong (got %.1f, "
                     "expected %.1f)\n",
                     rank, bad, N, got[0], expected);
        rc = 1;
    }

    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaFree(sendbuf));
    CUDA_CHECK(cudaFree(recvbuf));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));
    if (rank == 0) std::remove(idfile);
    return rc;
}
