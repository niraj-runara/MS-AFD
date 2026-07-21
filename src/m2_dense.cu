// MS-AFD · M2 (dense) — the fabric for a dense model.
//
// The dense analogue of the MoE fabric: micro-units carry no distinct weights
// (a dense model has one FFN per layer, not many experts), so "many units" comes
// from splitting *work* — the token batch — data-parallel across units. Every
// unit runs the *same* Llama-3 8B dense FFN on its own token tile. A unit's FFN
// is bit-for-bit the M0/M1 FFN, so its per-beat p99/p50 is directly comparable
// to M1 — only now the tile arrives over NVLink and fans out via CUDA IPC.
//
//   aside (GPU 0, NCCL rank 0): each beat, scatter a tile-batch to each of the H
//                               F-side GPUs (M2N), then gather all results back.
//   hub   (GPU g, NCCL rank g): recv this GPU's batch into an IPC-shared buffer,
//                               signal the local units, wait, send output back.
//   unit  (GPU g, MPS process): open the shared buffers via CUDA IPC, run the
//                               dense FFN (graph-captured) on its tile, signal.
//
// NCCL comm = 1 + H ranks (one per GPU — NCCL allows only one rank per GPU, see
// spike S1). Each GPU's hub is that rank and fans out to its units over CUDA IPC
// (spike S1.5). H=1 is the 2-GPU "vertical" case; H>1 is the M2N fabric.
//
// Headline: aside round-trip beat p99/p50 (whole fabric) and each unit's FFN
// compute p99/p50 (M1-comparable). Also supports a long soak via a big [beats].
//
// Usage:
//   m2_dense aside <H> <nunits> <idfile> [beats] [out.csv]
//   m2_dense hub   <rank> <H> <nunits> <idfile> <ctrl> <hdir> [beats] [out.csv]
//   m2_dense unit  <id> <nunits> <ctrl> <hdir> <device> [beats] [out.csv]
// Env: MSAFD_ARENA_MB sizes each unit's static HBM arena (default 768).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nccl.h>

#include "arena.h"
#include "ffn.h"

using namespace msafd;

#define MSAFD_NCCL_CHECK(call)                                                  \
    do {                                                                       \
        ncclResult_t r__ = (call);                                            \
        if (r__ != ncclSuccess) {                                             \
            std::fprintf(stderr, "[nccl] %s failed at %s:%d: %s\n", #call,    \
                         __FILE__, __LINE__, ncclGetErrorString(r__));        \
            std::abort();                                                     \
        }                                                                    \
    } while (0)

static const int kMaxUnits = 64;
static const int kWarmup   = 50;

// Cross-process control block (mmap'd file): the hub bumps `epoch`, each unit
// runs its FFN and writes its `epoch` into `done[id]`; `ready[id]` is the
// unit->hub "attached" handshake; `stop` tears the fabric down.
struct Control {
    volatile int epoch;
    volatile int stop;
    volatile int ready[kMaxUnits];
    volatile int done[kMaxUnits];
    int nunits;
};

// Blocking sync + sched_yield is mandatory for a many-process MPS fleet: with
// CUDA's default *spinning* sync, dozens of spinning threads starve each other
// even on a big host (spike S1.5 saw a 24x blowup at 48 units). See findings.md.
static void set_blocking_sync() {
    cudaError_t e = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (e != cudaSuccess && e != cudaErrorSetOnActiveProcess) MSAFD_CUDA_CHECK(e);
}

static Control* map_control(const char* path, bool create, int nunits) {
    int flags = create ? (O_CREAT | O_RDWR | O_TRUNC) : O_RDWR;
    int fd = open(path, flags, 0666);
    if (fd < 0) { std::perror("[m2] open ctrl"); std::exit(1); }
    if (create) {
        if (ftruncate(fd, sizeof(Control)) != 0) { std::perror("ftruncate"); std::exit(1); }
    } else {
        struct stat st;
        for (int t = 0;; ++t) {
            if (fstat(fd, &st) == 0 && st.st_size >= (off_t)sizeof(Control)) break;
            if (t > 1000) { std::fprintf(stderr, "[m2] ctrl size timeout\n"); std::exit(1); }
            usleep(20000);
        }
    }
    void* p = mmap(nullptr, sizeof(Control), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { std::perror("[m2] mmap"); std::exit(1); }
    close(fd);
    Control* c = static_cast<Control*>(p);
    if (create) { std::memset(p, 0, sizeof(Control)); c->nunits = nunits; }
    return c;
}

static void write_atomic(const std::string& path, const void* data, size_t n) {
    std::string tmp = path + ".tmp";
    { std::ofstream f(tmp, std::ios::binary);
      f.write(static_cast<const char*>(data), n); }
    std::rename(tmp.c_str(), path.c_str());
}

static bool read_exact(const std::string& path, void* data, size_t n) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || (size_t)f.tellg() < n) return false;
    f.seekg(0);
    f.read(static_cast<char*>(data), n);
    return true;
}

static void exchange_id(int rank, const char* idfile, ncclUniqueId* id) {
    if (rank == 0) {
        MSAFD_NCCL_CHECK(ncclGetUniqueId(id));
        write_atomic(idfile, id, sizeof(*id));
    } else {
        for (int t = 0;; ++t) {
            if (read_exact(idfile, id, sizeof(*id))) return;
            if (t > 2000) { std::fprintf(stderr, "[m2] id timeout\n"); std::exit(1); }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
}

static void report(const char* tag, const char* csv, std::vector<float> ms) {
    if (csv) {
        FILE* f = std::fopen(csv, "w");
        std::fprintf(f, "iter,ms\n");
        for (size_t i = 0; i < ms.size(); ++i) std::fprintf(f, "%zu,%.6f\n", i, ms[i]);
        std::fclose(f);
    }
    if (ms.empty()) { std::printf("[%s] no samples\n", tag); return; }
    std::sort(ms.begin(), ms.end());
    auto pct = [&](double p) { return ms[(size_t)(p * (ms.size() - 1))]; };
    std::printf("[%s] %s%sp50=%.4f ms  p99=%.4f ms  p99.9=%.4f ms  p99/p50=%.3f\n",
                tag, csv ? csv : "", csv ? " | " : "",
                pct(0.50), pct(0.99), pct(0.999), pct(0.99) / pct(0.50));
}

// ----------------------------------------------------------------- aside -----
static int run_aside(int H, int nunits, const char* idfile, int beats,
                     const char* csv) {
    MSAFD_CUDA_CHECK(cudaSetDevice(0));
    set_blocking_sync();
    FfnConfig cfg;
    const size_t seg   = (size_t)cfg.tokens * cfg.d_model * nunits;  // per-F-GPU batch
    const size_t total = seg * H;
    const size_t bytes = total * sizeof(__half);

    ncclUniqueId id; exchange_id(0, idfile, &id);
    ncclComm_t comm; MSAFD_NCCL_CHECK(ncclCommInitRank(&comm, 1 + H, id, 0));
    cudaStream_t stream; MSAFD_CUDA_CHECK(cudaStreamCreate(&stream));

    __half *tok, *res;
    MSAFD_CUDA_CHECK(cudaMalloc(&tok, bytes));
    MSAFD_CUDA_CHECK(cudaMalloc(&res, bytes));
    std::vector<__half> h(total);
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (auto& v : h) v = __float2half(dist(rng));
    MSAFD_CUDA_CHECK(cudaMemcpy(tok, h.data(), bytes, cudaMemcpyHostToDevice));
    MSAFD_CUDA_CHECK(cudaMemset(res, 0, bytes));

    std::vector<float> ms(beats);
    std::printf("[aside] GPU0 rank0: scatter to %d F-side GPU(s) x %d units, %d beats\n",
                H, nunits, beats);
    for (int i = 0; i < kWarmup + beats; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        MSAFD_NCCL_CHECK(ncclGroupStart());              // scatter to all F-GPUs
        for (int g = 0; g < H; ++g)
            MSAFD_NCCL_CHECK(ncclSend(tok + (size_t)g * seg, seg, ncclFloat16,
                                      g + 1, comm, stream));
        MSAFD_NCCL_CHECK(ncclGroupEnd());
        MSAFD_NCCL_CHECK(ncclGroupStart());              // gather from all F-GPUs
        for (int g = 0; g < H; ++g)
            MSAFD_NCCL_CHECK(ncclRecv(res + (size_t)g * seg, seg, ncclFloat16,
                                      g + 1, comm, stream));
        MSAFD_NCCL_CHECK(ncclGroupEnd());
        MSAFD_CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        if (i >= kWarmup)
            ms[i - kWarmup] = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }

    std::vector<__half> ho(total);
    MSAFD_CUDA_CHECK(cudaMemcpy(ho.data(), res, bytes, cudaMemcpyDeviceToHost));
    size_t nz = 0; for (auto& v : ho) if (__half2float(v) != 0.0f) ++nz;
    std::printf("[aside] gathered %zu/%zu non-zero elements from the fabric\n", nz, total);
    report("aside fabric beat", csv, ms);

    MSAFD_CUDA_CHECK(cudaFree(tok)); MSAFD_CUDA_CHECK(cudaFree(res));
    MSAFD_CUDA_CHECK(cudaStreamDestroy(stream));
    ncclCommDestroy(comm);
    std::remove(idfile);
    return 0;
}

// ------------------------------------------------------------------- hub -----
static int run_hub(int rank, int H, int nunits, const char* idfile,
                   const char* ctrl, const char* hdir, int beats,
                   const char* csv) {
    MSAFD_CUDA_CHECK(cudaSetDevice(rank));   // GPU g == rank g
    set_blocking_sync();
    FfnConfig cfg;
    const size_t seg   = (size_t)cfg.tokens * cfg.d_model * nunits;
    const size_t bytes = seg * sizeof(__half);

    ncclUniqueId id; exchange_id(rank, idfile, &id);
    ncclComm_t comm; MSAFD_NCCL_CHECK(ncclCommInitRank(&comm, 1 + H, id, rank));
    cudaStream_t stream; MSAFD_CUDA_CHECK(cudaStreamCreate(&stream));

    __half *in, *out;
    MSAFD_CUDA_CHECK(cudaMalloc(&in, bytes));
    MSAFD_CUDA_CHECK(cudaMalloc(&out, bytes));
    MSAFD_CUDA_CHECK(cudaMemset(out, 0, bytes));
    cudaIpcMemHandle_t hin, hout;
    MSAFD_CUDA_CHECK(cudaIpcGetMemHandle(&hin, in));
    MSAFD_CUDA_CHECK(cudaIpcGetMemHandle(&hout, out));
    write_atomic(std::string(hdir) + "/input.ipc", &hin, sizeof(hin));
    write_atomic(std::string(hdir) + "/output.ipc", &hout, sizeof(hout));

    Control* c = map_control(ctrl, true, nunits);
    std::printf("[hub r%d] GPU%d: %d units; waiting for them to attach...\n",
                rank, rank, nunits);
    for (;;) {
        int r = 0; for (int e = 0; e < nunits; ++e) r += c->ready[e];
        if (r == nunits) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::printf("[hub r%d] all units ready; serving %d beats\n", rank, beats);

    std::vector<float> ms(beats);
    int ep = 0;
    for (int i = 0; i < kWarmup + beats; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        // 1) receive the batch into the IPC-shared input buffer
        MSAFD_NCCL_CHECK(ncclGroupStart());
        MSAFD_NCCL_CHECK(ncclRecv(in, seg, ncclFloat16, 0, comm, stream));
        MSAFD_NCCL_CHECK(ncclGroupEnd());
        MSAFD_CUDA_CHECK(cudaStreamSynchronize(stream));  // data visible before units read
        // 2) fan out to local units, wait for all to finish this epoch
        ++ep; c->epoch = ep;
        for (;;) {
            int dn = 0; for (int e = 0; e < nunits; ++e) dn += (c->done[e] == ep);
            if (dn == nunits) break;
            sched_yield();
        }
        // 3) send the gathered output back to the A-side
        MSAFD_NCCL_CHECK(ncclGroupStart());
        MSAFD_NCCL_CHECK(ncclSend(out, seg, ncclFloat16, 0, comm, stream));
        MSAFD_NCCL_CHECK(ncclGroupEnd());
        MSAFD_CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        if (i >= kWarmup)
            ms[i - kWarmup] = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    c->stop = 1;

    report((std::string("hub r") + std::to_string(rank) + " beat").c_str(), csv, ms);
    MSAFD_CUDA_CHECK(cudaFree(in)); MSAFD_CUDA_CHECK(cudaFree(out));
    MSAFD_CUDA_CHECK(cudaStreamDestroy(stream));
    ncclCommDestroy(comm);
    return 0;
}

// ------------------------------------------------------------------ unit -----
static int run_unit(int id, int nunits, const char* ctrl, const char* hdir,
                    int device, int beats, const char* csv) {
    MSAFD_CUDA_CHECK(cudaSetDevice(device));   // this unit's F-side GPU
    set_blocking_sync();
    const char* pct = std::getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");

    cudaIpcMemHandle_t hin, hout;
    for (int t = 0;; ++t) {
        if (read_exact(std::string(hdir) + "/input.ipc", &hin, sizeof(hin)) &&
            read_exact(std::string(hdir) + "/output.ipc", &hout, sizeof(hout))) break;
        if (t > 2000) { std::fprintf(stderr, "[unit %d] IPC timeout\n", id); return 1; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    void *inbase, *outbase;
    MSAFD_CUDA_CHECK(cudaIpcOpenMemHandle(&inbase, hin, cudaIpcMemLazyEnablePeerAccess));
    MSAFD_CUDA_CHECK(cudaIpcOpenMemHandle(&outbase, hout, cudaIpcMemLazyEnablePeerAccess));

    FfnConfig cfg;
    const size_t slice = (size_t)cfg.tokens * cfg.d_model;  // this unit's tile
    __half* myin  = static_cast<__half*>(inbase)  + (size_t)id * slice;
    __half* myout = static_cast<__half*>(outbase) + (size_t)id * slice;

    std::size_t arena_mb =
        std::getenv("MSAFD_ARENA_MB") ? std::atoll(std::getenv("MSAFD_ARENA_MB")) : 768;
    Arena arena(arena_mb << 20);
    Ffn   ffn(cfg, arena);
    cudaStream_t stream; MSAFD_CUDA_CHECK(cudaStreamCreate(&stream));

    // capture the dense FFN into a replayable graph (bit-identical to M0/M1)
    ffn.forward(myin, myout, stream);
    MSAFD_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph; cudaGraphExec_t exec;
    MSAFD_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ffn.forward(myin, myout, stream);
    MSAFD_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    MSAFD_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

    cudaEvent_t s0, s1;
    MSAFD_CUDA_CHECK(cudaEventCreate(&s0));
    MSAFD_CUDA_CHECK(cudaEventCreate(&s1));

    Control* c = map_control(ctrl, false, nunits);
    std::printf("[unit %d] attached on GPU%d (MPS thread%%=%s), arena=%zu MB\n",
                id, device, pct ? pct : "unset", arena_mb);
    c->ready[id] = 1;

    // Per-unit FFN compute times (M1-comparable). Skip the hub's warmup beats.
    std::vector<float> ms;
    if (beats > 0) ms.reserve(beats);
    int last = 0, seen = 0;
    for (;;) {
        while (c->epoch == last && !c->stop) sched_yield();
        if (c->stop) break;
        last = c->epoch;
        MSAFD_CUDA_CHECK(cudaEventRecord(s0, stream));
        MSAFD_CUDA_CHECK(cudaGraphLaunch(exec, stream));
        MSAFD_CUDA_CHECK(cudaEventRecord(s1, stream));
        MSAFD_CUDA_CHECK(cudaStreamSynchronize(stream));
        c->done[id] = last;
        if (seen++ >= kWarmup) {
            float t; MSAFD_CUDA_CHECK(cudaEventElapsedTime(&t, s0, s1));
            ms.push_back(t);
        }
    }

    report((std::string("unit ") + std::to_string(id) + " ffn").c_str(), csv, ms);

    MSAFD_CUDA_CHECK(cudaGraphExecDestroy(exec));
    MSAFD_CUDA_CHECK(cudaGraphDestroy(graph));
    MSAFD_CUDA_CHECK(cudaEventDestroy(s0));
    MSAFD_CUDA_CHECK(cudaEventDestroy(s1));
    MSAFD_CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage:\n"
            "  %s aside <H> <nunits> <idfile> [beats] [out.csv]\n"
            "  %s hub   <rank> <H> <nunits> <idfile> <ctrl> <hdir> [beats] [out.csv]\n"
            "  %s unit  <id> <nunits> <ctrl> <hdir> <device> [beats] [out.csv]\n",
            argv[0], argv[0], argv[0]);
        return 2;
    }
    const std::string role = argv[1];
    if (role == "aside") {
        if (argc < 5) return 2;
        const int   H      = std::atoi(argv[2]);
        const int   nunits = std::atoi(argv[3]);
        const char* idfile = argv[4];
        const int   beats  = (argc > 5) ? std::atoi(argv[5]) : 10000;
        const char* csv    = (argc > 6) ? argv[6] : "results/m2_dense/aside.csv";
        return run_aside(H, nunits, idfile, beats, csv);
    } else if (role == "hub") {
        if (argc < 8) return 2;
        const int   rank   = std::atoi(argv[2]);
        const int   H      = std::atoi(argv[3]);
        const int   nunits = std::atoi(argv[4]);
        const char* idfile = argv[5];
        const char* ctrl   = argv[6];
        const char* hdir   = argv[7];
        const int   beats  = (argc > 8) ? std::atoi(argv[8]) : 10000;
        std::string dflt = "results/m2_dense/hub_r" + std::to_string(rank) + ".csv";
        const char* csv  = (argc > 9) ? argv[9] : dflt.c_str();
        return run_hub(rank, H, nunits, idfile, ctrl, hdir, beats, csv);
    } else if (role == "unit") {
        if (argc < 7) return 2;
        const int   id     = std::atoi(argv[2]);
        const int   nunits = std::atoi(argv[3]);
        const char* ctrl   = argv[4];
        const char* hdir   = argv[5];
        const int   device = std::atoi(argv[6]);
        const int   beats  = (argc > 7) ? std::atoi(argv[7]) : 10000;
        const char* csv    = (argc > 8) ? argv[8] : nullptr;
        return run_unit(id, nunits, ctrl, hdir, device, beats, csv);
    }
    std::fprintf(stderr, "unknown role '%s'\n", role.c_str());
    return 2;
}
