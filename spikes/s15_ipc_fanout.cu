// MS-AFD · Spike S1.5 — intra-GPU fan-out via CUDA IPC (the M2 bridge).
//
// NCCL forbids >1 rank per GPU (see S1), so in M2 each F-side GPU has ONE NCCL
// rank ("hub") but many MPS-isolated expert micro-units. S1.5 proves the bridge
// between them, on a SINGLE GPU, with no NCCL in the way:
//
//   hub process        : owns shared input/output device buffers (cudaMalloc),
//                        exports them via CUDA IPC, and each beat signals the
//                        experts and waits for them to finish.
//   N expert processes : MPS-capped micro-units. Each opens the shared buffers
//                        via IPC, runs its FFN (the M0/M1 Ffn, graph-captured)
//                        on its own slice, and signals done.
//
// Coordination is a tiny mmap'd shared-memory control block (epoch / done / stop)
// busy-polled across processes. Metric: the hub's per-beat p99/p50 — determinism
// of the whole fan-out — plus a check that the routed output is deterministic.
//
// Usage:
//   s15_ipc_fanout hub    <nexp> <ctrl_file> <handle_dir> [beats] [out.csv]
//   s15_ipc_fanout expert <id>   <nexp> <ctrl_file> <handle_dir>

#include <algorithm>
#include <chrono>
#include <cstdint>
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
#include <cuda_bf16.h>

#include "check.h"
#include "arena.h"
#include "ffn.h"

using namespace msafd;

static const int kMaxExp = 64;

// Shared-memory control block (mmap'd file, shared across all processes).
struct Control {
    volatile int epoch;            // hub bumps to start a beat
    volatile int stop;             // hub sets to end the run
    volatile int ready[kMaxExp];   // expert e sets 1 once attached + captured
    volatile int done[kMaxExp];    // expert e sets = epoch when its beat is done
    int nexp;
};

// Make cudaStreamSynchronize block (yield the CPU) instead of busy-polling, so
// N expert processes don't thrash the scheduler. Must run before the context
// exists; tolerate the "already active" error.
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

// -------------------------------------------------------------------- hub -----
static int run_hub(int nexp, const char* ctrl, const char* hdir, int beats,
                   const char* csv) {
    CUDA_CHECK(cudaSetDevice(0));
    set_blocking_sync();
    FfnConfig cfg;
    const int T = cfg.tokens, D = cfg.d_model;
    const size_t slice = (size_t)T * D;
    const size_t total = slice * nexp;
    const size_t bytes = total * sizeof(__nv_bfloat16);

    __nv_bfloat16 *in, *out;
    CUDA_CHECK(cudaMalloc(&in, bytes));
    CUDA_CHECK(cudaMalloc(&out, bytes));

    std::vector<float> h(total);
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h) v = dist(rng);
    std::vector<__nv_bfloat16> hbf(total);
    for (size_t i = 0; i < total; ++i) hbf[i] = __float2bfloat16(h[i]);
    CUDA_CHECK(cudaMemcpy(in, hbf.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(out, 0, bytes));

    cudaIpcMemHandle_t hin, hout;
    CUDA_CHECK(cudaIpcGetMemHandle(&hin, in));
    CUDA_CHECK(cudaIpcGetMemHandle(&hout, out));
    write_atomic(std::string(hdir) + "/input.ipc", &hin, sizeof(hin));
    write_atomic(std::string(hdir) + "/output.ipc", &hout, sizeof(hout));

    Control* c = map_control(ctrl, true, nexp);
    std::printf("[hub] %d experts, tile=%dx%d; IPC handles published; waiting...\n",
                nexp, T, D);
    for (;;) {
        int r = 0;
        for (int e = 0; e < nexp; ++e) r += c->ready[e];
        if (r == nexp) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::printf("[hub] all experts ready; running %d beats\n", beats);

    auto beat = [&](int ep) {
        c->epoch = ep;
        for (;;) {
            int dn = 0;
            for (int e = 0; e < nexp; ++e) dn += (c->done[e] == ep);
            if (dn == nexp) break;
            sched_yield();  // don't hard-spin against the expert processes
        }
    };
    for (int w = 1; w <= 50; ++w) beat(w);  // warmup

    std::vector<float> ms(beats);
    int ep = 50;
    for (int b = 0; b < beats; ++b) {
        auto t0 = std::chrono::high_resolution_clock::now();
        beat(++ep);
        auto t1 = std::chrono::high_resolution_clock::now();
        ms[b] = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    c->stop = 1;  // release experts (they see stop while waiting)

    std::vector<__nv_bfloat16> ho(total);
    CUDA_CHECK(cudaMemcpy(ho.data(), out, bytes, cudaMemcpyDeviceToHost));
    size_t nz = 0;
    for (auto& v : ho) if (__bfloat162float(v) != 0.0f) ++nz;
    std::printf("[hub] output non-zero: %zu/%zu elements\n", nz, total);

    FILE* f = std::fopen(csv, "w");
    std::fprintf(f, "iter,ms\n");
    for (int b = 0; b < beats; ++b) std::fprintf(f, "%d,%.6f\n", b, ms[b]);
    std::fclose(f);
    std::sort(ms.begin(), ms.end());
    auto pct = [&](double p) { return ms[(size_t)(p * (beats - 1))]; };
    std::printf("[hub] wrote %s | fan-out beat p50=%.4f ms  p99=%.4f ms  "
                "p99/p50=%.3f\n", csv, pct(0.50), pct(0.99), pct(0.99) / pct(0.50));

    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    return 0;
}

// ----------------------------------------------------------------- expert -----
static int run_expert(int id, int nexp, const char* ctrl, const char* hdir) {
    CUDA_CHECK(cudaSetDevice(0));
    set_blocking_sync();
    const char* pct = std::getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");

    cudaIpcMemHandle_t hin, hout;
    for (int t = 0;; ++t) {
        if (read_exact(std::string(hdir) + "/input.ipc", &hin, sizeof(hin)) &&
            read_exact(std::string(hdir) + "/output.ipc", &hout, sizeof(hout)))
            break;
        if (t > 1000) { std::fprintf(stderr, "expert %d: IPC handle timeout\n", id); return 1; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    void *inbase, *outbase;
    CUDA_CHECK(cudaIpcOpenMemHandle(&inbase, hin, cudaIpcMemLazyEnablePeerAccess));
    CUDA_CHECK(cudaIpcOpenMemHandle(&outbase, hout, cudaIpcMemLazyEnablePeerAccess));

    FfnConfig cfg;
    const int T = cfg.tokens, D = cfg.d_model;
    const size_t slice = (size_t)T * D;
    __nv_bfloat16* myin  = static_cast<__nv_bfloat16*>(inbase) + (size_t)id * slice;
    __nv_bfloat16* myout = static_cast<__nv_bfloat16*>(outbase) + (size_t)id * slice;

    Arena arena;
    Ffn ffn(cfg, arena);
    const size_t ws_bytes = 4ull << 20;
    void* ws = arena.alloc(ws_bytes, "cublas_ws");
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetWorkspace(handle, ws, ws_bytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    ffn.forward(myin, myout, handle, stream);          // pre-warm cuBLAS
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph; cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ffn.forward(myin, myout, handle, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));

    Control* c = map_control(ctrl, false, nexp);
    std::printf("[expert %d] attached (MPS thread%%=%s); ready\n", id,
                pct ? pct : "unset");
    c->ready[id] = 1;

    int last = 0;
    for (;;) {
        while (c->epoch == last && !c->stop) sched_yield();  // yield between beats
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
    if (argc < 5) {
        std::fprintf(stderr,
            "usage:\n"
            "  %s hub    <nexp> <ctrl_file> <handle_dir> [beats] [out.csv]\n"
            "  %s expert <id>   <nexp> <ctrl_file> <handle_dir>\n",
            argv[0], argv[0]);
        return 2;
    }
    const std::string role = argv[1];
    if (role == "hub") {
        const int nexp = std::atoi(argv[2]);
        const char* ctrl = argv[3];
        const char* hdir = argv[4];
        const int beats = (argc > 5) ? std::atoi(argv[5]) : 10000;
        const char* csv = (argc > 6) ? argv[6] : "results/s15/beat.csv";
        return run_hub(nexp, ctrl, hdir, beats, csv);
    } else if (role == "expert") {
        const int id = std::atoi(argv[2]);
        const int nexp = std::atoi(argv[3]);
        const char* ctrl = argv[4];
        const char* hdir = argv[5];
        return run_expert(id, nexp, ctrl, hdir);
    }
    std::fprintf(stderr, "unknown role '%s'\n", role.c_str());
    return 2;
}
