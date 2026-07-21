// MS-AFD · M4 (G1) — the live-fabric coordinator (A-side), loaded into HF.
//
// A C-ABI shared library HF calls via ctypes. It owns the IPC input/output tiles
// and the control block; each patched LlamaMLP.forward calls msafd_live_ffn(),
// which ships the hidden state to the layer's live unit process over CUDA IPC,
// waits for it, and returns the result. HF runs attention/KV/generation; every
// FFN is served by a separate MPS-capped unit process — the live fabric.
//
// extern "C" so Python can call it with torch .data_ptr()s.

#include <cstdio>
#include <cstdlib>
#include <sched.h>
#include <unistd.h>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "live_common.h"

using namespace msafd;
using bf16 = __nv_bfloat16;

namespace {
LiveControl* g_ctrl = nullptr;
bf16*        g_in   = nullptr;
bf16*        g_out  = nullptr;
int          g_D    = 0;
int          g_nlayers = 0;
}  // namespace

extern "C" {

// Create the IPC buffers + control block, publish handles, wait for all units.
void msafd_live_init(int n_layers, int D, int T_max, int device,
                     const char* hdir, const char* ctrl) {
    LIVE_CUDA_CHECK(cudaSetDevice(device));
    g_D = D; g_nlayers = n_layers;
    LIVE_CUDA_CHECK(cudaMalloc(&g_in,  (size_t)T_max * D * sizeof(bf16)));
    LIVE_CUDA_CHECK(cudaMalloc(&g_out, (size_t)T_max * D * sizeof(bf16)));

    cudaIpcMemHandle_t hin, hout;
    LIVE_CUDA_CHECK(cudaIpcGetMemHandle(&hin, g_in));
    LIVE_CUDA_CHECK(cudaIpcGetMemHandle(&hout, g_out));
    live_write_handle((std::string(hdir) + "/in.ipc").c_str(), &hin, sizeof(hin));
    live_write_handle((std::string(hdir) + "/out.ipc").c_str(), &hout, sizeof(hout));

    g_ctrl = live_map_control(ctrl, true, n_layers);
    std::fprintf(stderr, "[coord] waiting for %d units...\n", n_layers);
    for (int t = 0;; ++t) {
        int r = 0; for (int L = 0; L < n_layers; ++L) r += g_ctrl->ready[L];
        if (r == n_layers) break;
        if (t % 100 == 0) std::fprintf(stderr, "[coord] units ready: %d/%d\n", r, n_layers);
        usleep(50000);
    }
    std::fprintf(stderr, "[coord] all %d units ready — live fabric up\n", n_layers);
}

// Serve one layer's FFN through its live unit. hidden/out are device bf16 [T,D].
void msafd_live_ffn(int layer, const void* hidden, void* out, int T) {
    LIVE_CUDA_CHECK(cudaMemcpy(g_in, hidden, (size_t)T * g_D * sizeof(bf16),
                               cudaMemcpyDeviceToDevice));
    g_ctrl->T = T;
    g_ctrl->active_layer = layer;
    int e = ++g_ctrl->epoch;
    while (g_ctrl->done_epoch != e) sched_yield();
    g_ctrl->active_layer = -1;
    LIVE_CUDA_CHECK(cudaMemcpy(out, g_out, (size_t)T * g_D * sizeof(bf16),
                               cudaMemcpyDeviceToDevice));
}

void msafd_live_stop() {
    if (g_ctrl) g_ctrl->stop = 1;
}

}  // extern "C"
