// MS-AFD · M3-mini through the real fabric — correctness through hub + IPC + MPS.
//
// Same real Qwen3 MoE layer as m3_layer, but instead of one process, it runs
// through the actual fabric machinery on a single GPU: a hub lays out each active
// expert's capacity-padded real tokens in a CUDA-IPC-shared buffer; one MPS
// expert process per active expert opens the buffer, loads its REAL weights, runs
// its FFN (graph-captured) each beat; the hub gathers, combines per token with
// the router weights, and checks rel-L2 vs HF's moe_out. Also times the fan-out
// beat (determinism sanity — expected to match M2/S1.5, since capacity-padding
// fixes the tile shape regardless of routing).
//
// Usage:
//   m3_fabric hub    <ref_dir> [beats]
//   m3_fabric expert <proc_id> <ref_dir>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

#include "check.h"
#include "arena.h"
#include "ffn.h"

using namespace msafd;

static const int kMaxExp = 256;

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

static bool read_exact_file(const std::string& path, void* data, size_t n) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || (size_t)f.tellg() < n) return false;
    f.seekg(0);
    f.read(static_cast<char*>(data), n);
    return true;
}

template <typename T>
static std::vector<T> read_bin(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<T> v(count);
    f.read(reinterpret_cast<char*>(v.data()), count * sizeof(T));
    return v;
}

struct Meta { int T, D, F, nexp_total, topk; };
static Meta read_meta(const std::string& dir) {
    Meta m{};
    std::ifstream f(dir + "/meta.txt");
    if (!f || !(f >> m.T >> m.D >> m.F >> m.nexp_total >> m.topk)) {
        std::fprintf(stderr, "cannot read %s/meta.txt\n", dir.c_str()); std::exit(1);
    }
    return m;
}
static std::vector<int> read_active(const std::string& dir) {
    std::vector<int> a; std::ifstream f(dir + "/active_experts.txt"); int e;
    while (f >> e) a.push_back(e);
    return a;
}
static int read_capacity(const std::string& dir) {
    std::ifstream f(dir + "/capacity.txt"); int c = 0; f >> c; return c;
}

// -------------------------------------------------------------------- hub -----
static int run_hub(const std::string& dir, int beats) {
    CUDA_CHECK(cudaSetDevice(0));
    set_blocking_sync();
    Meta m = read_meta(dir);
    std::vector<int> active = read_active(dir);
    const int C = read_capacity(dir);
    const int nexp = (int)active.size();
    const size_t slice = (size_t)C * m.D;
    const size_t total = slice * nexp;

    auto hidden = read_bin<__nv_bfloat16>(dir + "/hidden_in.bin", (size_t)m.T * m.D);
    auto refout = read_bin<__nv_bfloat16>(dir + "/moe_out.bin",   (size_t)m.T * m.D);
    auto idx    = read_bin<int32_t>(dir + "/topk_idx.bin", (size_t)m.T * m.topk);
    auto wgt    = read_bin<float>(dir + "/topk_w.bin",     (size_t)m.T * m.topk);

    // tokens routed to each active expert (slot order = fill order).
    std::vector<std::vector<int>> toks(nexp);
    for (int i = 0; i < nexp; ++i)
        for (int t = 0; t < m.T; ++t)
            for (int k = 0; k < m.topk; ++k)
                if (idx[(size_t)t * m.topk + k] == active[i]) toks[i].push_back(t);

    __nv_bfloat16 *in, *out;
    CUDA_CHECK(cudaMalloc(&in, total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&out, total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(out, 0, total * sizeof(__nv_bfloat16)));

    // Lay out each expert's capacity-padded real tokens into its slice.
    std::vector<__nv_bfloat16> h_in(total, __float2bfloat16(0.0f));
    for (int i = 0; i < nexp; ++i)
        for (size_t s = 0; s < toks[i].size(); ++s)
            for (int j = 0; j < m.D; ++j)
                h_in[i * slice + s * m.D + j] = hidden[(size_t)toks[i][s] * m.D + j];
    CUDA_CHECK(cudaMemcpy(in, h_in.data(), total * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    cudaIpcMemHandle_t hin, hout;
    CUDA_CHECK(cudaIpcGetMemHandle(&hin, in));
    CUDA_CHECK(cudaIpcGetMemHandle(&hout, out));
    write_atomic(dir + "/input.ipc", &hin, sizeof(hin));
    write_atomic(dir + "/output.ipc", &hout, sizeof(hout));

    const std::string ctrl = dir + "/m3_ctrl";
    Control* c = map_control(ctrl.c_str(), true, nexp);
    std::printf("[hub] %d active experts, C=%d, tile=%dx%d; waiting...\n", nexp, C, C, m.D);
    for (;;) { int r = 0; for (int e = 0; e < nexp; ++e) r += c->ready[e];
               if (r == nexp) break; std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
    std::printf("[hub] all experts ready; running %d beats\n", beats);

    auto beat = [&](int ep) {
        c->epoch = ep;
        for (;;) { int dn = 0; for (int e = 0; e < nexp; ++e) dn += (c->done[e] == ep);
                   if (dn == nexp) break; sched_yield(); }
    };
    for (int w = 1; w <= 50; ++w) beat(w);
    std::vector<float> ms(beats);
    int ep = 50;
    for (int b = 0; b < beats; ++b) {
        auto t0 = std::chrono::high_resolution_clock::now();
        beat(++ep);
        auto t1 = std::chrono::high_resolution_clock::now();
        ms[b] = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    c->stop = 1;

    // Gather + router-weighted combine, compare to HF reference.
    std::vector<__nv_bfloat16> h_out(total);
    CUDA_CHECK(cudaMemcpy(h_out.data(), out, total * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    std::vector<double> comb((size_t)m.T * m.D, 0.0);
    for (int i = 0; i < nexp; ++i) {
        int e = active[i];
        for (size_t s = 0; s < toks[i].size(); ++s) {
            int t = toks[i][s];
            float w = 0.0f;
            for (int k = 0; k < m.topk; ++k)
                if (idx[(size_t)t * m.topk + k] == e) { w = wgt[(size_t)t * m.topk + k]; break; }
            for (int j = 0; j < m.D; ++j)
                comb[(size_t)t * m.D + j] += (double)w * __bfloat162float(h_out[i * slice + s * m.D + j]);
        }
    }
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < (size_t)m.T * m.D; ++i) {
        double r = __bfloat162float(refout[i]), d = comb[i] - r;
        num += d * d; den += r * r;
    }
    double rel_l2 = std::sqrt(num) / (std::sqrt(den) + 1e-12);

    std::sort(ms.begin(), ms.end());
    auto pct = [&](double p) { return ms[(size_t)(p * (beats - 1))]; };
    std::printf("[hub] fan-out beat p50=%.4f ms  p99=%.4f ms  p99/p50=%.3f\n",
                pct(0.50), pct(0.99), pct(0.99) / pct(0.50));
    std::printf("[hub] rel-L2 vs HF moe_out = %.5f  ->  %s\n", rel_l2,
                rel_l2 <= 0.05 ? "CORRECT through the fabric" : "MISMATCH");

    CUDA_CHECK(cudaFree(in)); CUDA_CHECK(cudaFree(out));
    return rel_l2 <= 0.05 ? 0 : 1;
}

// ----------------------------------------------------------------- expert -----
static int run_expert(int proc_id, const std::string& dir) {
    CUDA_CHECK(cudaSetDevice(0));
    set_blocking_sync();
    Meta m = read_meta(dir);
    std::vector<int> active = read_active(dir);
    const int C = read_capacity(dir);
    const int nexp = (int)active.size();
    const int my_expert = active[proc_id];
    const size_t slice = (size_t)C * m.D;

    // Wait for the hub's IPC handles.
    cudaIpcMemHandle_t hin, hout;
    for (int t = 0;; ++t) {
        if (read_exact_file(dir + "/input.ipc", &hin, sizeof(hin)) &&
            read_exact_file(dir + "/output.ipc", &hout, sizeof(hout))) break;
        if (t > 2000) { std::fprintf(stderr, "expert %d: IPC timeout\n", proc_id); return 1; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    void *inbase, *outbase;
    CUDA_CHECK(cudaIpcOpenMemHandle(&inbase, hin, cudaIpcMemLazyEnablePeerAccess));
    CUDA_CHECK(cudaIpcOpenMemHandle(&outbase, hout, cudaIpcMemLazyEnablePeerAccess));
    __nv_bfloat16* myin  = static_cast<__nv_bfloat16*>(inbase) + (size_t)proc_id * slice;
    __nv_bfloat16* myout = static_cast<__nv_bfloat16*>(outbase) + (size_t)proc_id * slice;

    // Real weights for this expert.
    const size_t wgu = (size_t)m.D * m.F, wd = (size_t)m.F * m.D;
    auto g = read_bin<__nv_bfloat16>(dir + "/experts/e" + std::to_string(my_expert) + "_gate.bin", wgu);
    auto u = read_bin<__nv_bfloat16>(dir + "/experts/e" + std::to_string(my_expert) + "_up.bin",   wgu);
    auto d = read_bin<__nv_bfloat16>(dir + "/experts/e" + std::to_string(my_expert) + "_down.bin", wd);

    FfnConfig cfg; cfg.d_model = m.D; cfg.d_intermediate = m.F; cfg.tokens = C;
    Arena arena;
    Ffn ffn(cfg, arena);
    ffn.load_weights_bf16(g.data(), u.data(), d.data());
    const size_t ws = 4ull << 20;
    void* d_ws = arena.alloc(ws, "cublas_ws");
    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetWorkspace(handle, d_ws, ws));
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

    ffn.forward(myin, myout, handle, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph; cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ffn.forward(myin, myout, handle, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));

    const std::string ctrl = dir + "/m3_ctrl";
    Control* c = map_control(ctrl.c_str(), false, nexp);
    c->ready[proc_id] = 1;
    int last = 0;
    for (;;) {
        while (c->epoch == last && !c->stop) sched_yield();
        if (c->stop) break;
        last = c->epoch;
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        c->done[proc_id] = last;
    }
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage:\n  %s hub    <ref_dir> [beats]\n  %s expert <proc_id> <ref_dir>\n",
            argv[0], argv[0]);
        return 2;
    }
    const std::string role = argv[1];
    if (role == "hub") {
        const std::string dir = argv[2];
        const int beats = (argc > 3) ? std::atoi(argv[3]) : 2000;
        return run_hub(dir, beats);
    } else if (role == "expert") {
        const int proc_id = std::atoi(argv[2]);
        const std::string dir = argv[3];
        return run_expert(proc_id, dir);
    }
    std::fprintf(stderr, "unknown role '%s'\n", role.c_str());
    return 2;
}
