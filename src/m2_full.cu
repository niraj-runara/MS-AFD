// MS-AFD · M2-full — the fabric. A-side router scatters token tiles to H F-side
// GPUs (M2N) and gathers results; each F-side GPU is the M2-vertical stack
// (NCCL hub + IPC fan-out to N MPS expert processes).
//
//   aside  (GPU 0, NCCL rank 0): each beat, scatter a tile-batch to each of the
//                                H F-side GPUs, then gather all results back.
//   hub    (GPU g, NCCL rank g): recv this GPU's batch into an IPC-shared buffer,
//                                fan out to local experts, send the gathered
//                                output back to rank 0.  g = 1..H
//   expert (GPU g, MPS process): open shared buffers via IPC, run its FFN slice.
//
// NCCL comm = 1 + H ranks (one per GPU; NCCL allows one rank per GPU — see S1).
// Headline: aside round-trip beat p99/p50 — determinism of the whole fabric,
// gated by the slowest F-side GPU each beat. Also supports a long soak (beats).
//
// Usage:
//   m2_full aside  <H> <nexp> <idfile> [beats] [out.csv]
//   m2_full hub    <rank> <H> <nexp> <idfile> <ctrl> <hdir> [beats] [out.csv]
//   m2_full expert <id> <nexp> <ctrl> <hdir> <device>

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
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <nccl.h>

#include "check.h"
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

static const int kMaxExp = 64;

struct Control {
    volatile int epoch;
    volatile int stop;
    volatile int ready[kMaxExp];
    volatile int done[kMaxExp];
    int nexp;
};

static void set_blocking_sync() {
    cudaError_t e = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (e != cudaSuccess && e != cudaErrorSetOnActiveProcess) CUDA_CHECK(e);
}

static Control* map_control(const char* path, bool create, int nexp) {
    int flags = create ? (O_CREAT | O_RDWR | O_TRUNC) : O_RDWR;
    int fd = open(path, flags, 0666);
    if (fd < 0) { std::perror("open ctrl"); std::exit(1); }
    if (create) {
        if (ftruncate(fd, sizeof(Control)) != 0) { std::perror("ftruncate"); std::exit(1); }
    } else {
        struct stat st;
        for (int t = 0;; ++t) {
            if (fstat(fd, &st) == 0 && st.st_size >= (off_t)sizeof(Control)) break;
            if (t > 1000) { std::fprintf(stderr, "ctrl size timeout\n"); std::exit(1); }
            usleep(20000);
        }
    }
    void* p = mmap(nullptr, sizeof(Control), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
    close(fd);
    Control* c = static_cast<Control*>(p);
    if (create) { std::memset(p, 0, sizeof(Control)); c->nexp = nexp; }
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
        NCCL_CHECK(ncclGetUniqueId(id));
        write_atomic(idfile, id, sizeof(*id));
    } else {
        for (int t = 0;; ++t) {
            if (read_exact(idfile, id, sizeof(*id))) return;
            if (t > 2000) { std::fprintf(stderr, "id timeout\n"); std::exit(1); }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
}

static void report(const char* tag, const char* csv, std::vector<float> ms) {
    FILE* f = std::fopen(csv, "w");
    std::fprintf(f, "iter,ms\n");
    for (size_t i = 0; i < ms.size(); ++i) std::fprintf(f, "%zu,%.6f\n", i, ms[i]);
    std::fclose(f);
    std::sort(ms.begin(), ms.end());
    auto pct = [&](double p) { return ms[(size_t)(p * (ms.size() - 1))]; };
    std::printf("[%s] wrote %s | p50=%.4f ms  p99=%.4f ms  p99.9=%.4f ms  "
                "p99/p50=%.3f\n", tag, csv, pct(0.50), pct(0.99), pct(0.999),
                pct(0.99) / pct(0.50));
}

// ----------------------------------------------------------------- aside -----
static int run_aside(int H, int nexp, const char* idfile, int beats,
                     const char* csv) {
    CUDA_CHECK(cudaSetDevice(0));
    set_blocking_sync();
    FfnConfig cfg;
    const size_t seg = (size_t)cfg.tokens * cfg.d_model * nexp;  // per-F-GPU batch
    const size_t total = seg * H;
    const size_t bytes = total * sizeof(__nv_bfloat16);

    ncclUniqueId id; exchange_id(0, idfile, &id);
    ncclComm_t comm; NCCL_CHECK(ncclCommInitRank(&comm, 1 + H, id, 0));
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

    __nv_bfloat16 *tok, *res;
    CUDA_CHECK(cudaMalloc(&tok, bytes));
    CUDA_CHECK(cudaMalloc(&res, bytes));
    std::vector<float> h(total);
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h) v = dist(rng);
    std::vector<__nv_bfloat16> hbf(total);
    for (size_t i = 0; i < total; ++i) hbf[i] = __float2bfloat16(h[i]);
    CUDA_CHECK(cudaMemcpy(tok, hbf.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(res, 0, bytes));

    const int warmup = 50;
    std::vector<float> ms(beats);
    std::printf("[aside] GPU0 rank0: scatter to %d F-side GPUs x %d experts, "
                "%d beats\n", H, nexp, beats);
    for (int i = 0; i < warmup + beats; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        NCCL_CHECK(ncclGroupStart());                    // scatter to all F-GPUs
        for (int g = 0; g < H; ++g)
            NCCL_CHECK(ncclSend(tok + (size_t)g * seg, seg, ncclBfloat16,
                                g + 1, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        NCCL_CHECK(ncclGroupStart());                    // gather from all F-GPUs
        for (int g = 0; g < H; ++g)
            NCCL_CHECK(ncclRecv(res + (size_t)g * seg, seg, ncclBfloat16,
                                g + 1, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        if (i >= warmup)
            ms[i - warmup] = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }

    std::vector<__nv_bfloat16> ho(total);
    CUDA_CHECK(cudaMemcpy(ho.data(), res, bytes, cudaMemcpyDeviceToHost));
    size_t nz = 0; for (auto& v : ho) if (__bfloat162float(v) != 0.0f) ++nz;
    std::printf("[aside] gathered %zu/%zu non-zero elements from the fabric\n",
                nz, total);
    report("aside fabric beat", csv, ms);

    CUDA_CHECK(cudaFree(tok)); CUDA_CHECK(cudaFree(res));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));
    std::remove(idfile);
    return 0;
}

// ------------------------------------------------------------------- hub -----
static int run_hub(int rank, int H, int nexp, const char* idfile,
                   const char* ctrl, const char* hdir, int beats,
                   const char* csv) {
    CUDA_CHECK(cudaSetDevice(rank));   // GPU g == rank g
    set_blocking_sync();
    FfnConfig cfg;
    const size_t seg = (size_t)cfg.tokens * cfg.d_model * nexp;
    const size_t bytes = seg * sizeof(__nv_bfloat16);

    ncclUniqueId id; exchange_id(rank, idfile, &id);
    ncclComm_t comm; NCCL_CHECK(ncclCommInitRank(&comm, 1 + H, id, rank));
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

    __nv_bfloat16 *in, *out;
    CUDA_CHECK(cudaMalloc(&in, bytes));
    CUDA_CHECK(cudaMalloc(&out, bytes));
    CUDA_CHECK(cudaMemset(out, 0, bytes));
    cudaIpcMemHandle_t hin, hout;
    CUDA_CHECK(cudaIpcGetMemHandle(&hin, in));
    CUDA_CHECK(cudaIpcGetMemHandle(&hout, out));
    write_atomic(std::string(hdir) + "/input.ipc", &hin, sizeof(hin));
    write_atomic(std::string(hdir) + "/output.ipc", &hout, sizeof(hout));

    Control* c = map_control(ctrl, true, nexp);
    std::printf("[hub r%d] GPU%d: %d experts; waiting...\n", rank, rank, nexp);
    for (;;) {
        int r = 0; for (int e = 0; e < nexp; ++e) r += c->ready[e];
        if (r == nexp) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::printf("[hub r%d] all experts ready; serving %d beats\n", rank, beats);

    const int warmup = 50;
    std::vector<float> ms(beats);
    int ep = 0;
    for (int i = 0; i < warmup + beats; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclRecv(in, seg, ncclBfloat16, 0, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ++ep; c->epoch = ep;
        for (;;) {
            int dn = 0; for (int e = 0; e < nexp; ++e) dn += (c->done[e] == ep);
            if (dn == nexp) break;
            sched_yield();
        }
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(out, seg, ncclBfloat16, 0, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        if (i >= warmup)
            ms[i - warmup] = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    c->stop = 1;

    report((std::string("hub r") + std::to_string(rank) + " beat").c_str(), csv, ms);
    CUDA_CHECK(cudaFree(in)); CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));
    return 0;
}

// ---------------------------------------------------------------- expert -----
static int run_expert(int id, int nexp, const char* ctrl, const char* hdir,
                      int device) {
    CUDA_CHECK(cudaSetDevice(device));   // this expert's F-side GPU
    set_blocking_sync();
    const char* pct = std::getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");

    cudaIpcMemHandle_t hin, hout;
    for (int t = 0;; ++t) {
        if (read_exact(std::string(hdir) + "/input.ipc", &hin, sizeof(hin)) &&
            read_exact(std::string(hdir) + "/output.ipc", &hout, sizeof(hout))) break;
        if (t > 2000) { std::fprintf(stderr, "expert %d: IPC timeout\n", id); return 1; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    void *inbase, *outbase;
    CUDA_CHECK(cudaIpcOpenMemHandle(&inbase, hin, cudaIpcMemLazyEnablePeerAccess));
    CUDA_CHECK(cudaIpcOpenMemHandle(&outbase, hout, cudaIpcMemLazyEnablePeerAccess));

    FfnConfig cfg;
    const size_t slice = (size_t)cfg.tokens * cfg.d_model;
    __nv_bfloat16* myin  = static_cast<__nv_bfloat16*>(inbase) + (size_t)id * slice;
    __nv_bfloat16* myout = static_cast<__nv_bfloat16*>(outbase) + (size_t)id * slice;

    Arena arena;
    Ffn ffn(cfg, arena);
    const size_t ws_bytes = 4ull << 20;
    void* ws = arena.alloc(ws_bytes, "cublas_ws");
    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetWorkspace(handle, ws, ws_bytes));
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

    ffn.forward(myin, myout, handle, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph; cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ffn.forward(myin, myout, handle, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));

    Control* c = map_control(ctrl, false, nexp);
    std::printf("[expert %d] attached (MPS thread%%=%s)\n", id, pct ? pct : "unset");
    c->ready[id] = 1;

    int last = 0;
    for (;;) {
        while (c->epoch == last && !c->stop) sched_yield();
        if (c->stop) break;
        last = c->epoch;
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        c->done[id] = last;
    }

    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage:\n"
            "  %s aside  <H> <nexp> <idfile> [beats] [out.csv]\n"
            "  %s hub    <rank> <H> <nexp> <idfile> <ctrl> <hdir> [beats] [out.csv]\n"
            "  %s expert <id> <nexp> <ctrl> <hdir> <device>\n",
            argv[0], argv[0], argv[0]);
        return 2;
    }
    const std::string role = argv[1];
    if (role == "aside") {
        if (argc < 5) return 2;
        const int H = std::atoi(argv[2]);
        const int nexp = std::atoi(argv[3]);
        const char* idfile = argv[4];
        const int beats = (argc > 5) ? std::atoi(argv[5]) : 10000;
        const char* csv = (argc > 6) ? argv[6] : "results/m2_full/aside.csv";
        return run_aside(H, nexp, idfile, beats, csv);
    } else if (role == "hub") {
        if (argc < 8) return 2;
        const int rank = std::atoi(argv[2]);
        const int H = std::atoi(argv[3]);
        const int nexp = std::atoi(argv[4]);
        const char* idfile = argv[5];
        const char* ctrl = argv[6];
        const char* hdir = argv[7];
        const int beats = (argc > 8) ? std::atoi(argv[8]) : 10000;
        std::string default_csv = "results/m2_full/hub_r" + std::to_string(rank) + ".csv";
        const char* csv = (argc > 9) ? argv[9] : default_csv.c_str();
        return run_hub(rank, H, nexp, idfile, ctrl, hdir, beats, csv);
    } else if (role == "expert") {
        if (argc < 7) return 2;
        const int id = std::atoi(argv[2]);
        const int nexp = std::atoi(argv[3]);
        const char* ctrl = argv[4];
        const char* hdir = argv[5];
        const int device = std::atoi(argv[6]);
        return run_expert(id, nexp, ctrl, hdir, device);
    }
    std::fprintf(stderr, "unknown role '%s'\n", role.c_str());
    return 2;
}
