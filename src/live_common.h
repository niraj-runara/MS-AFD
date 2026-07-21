#pragma once
// MS-AFD · M4 (G1) — shared plumbing for the LIVE dense fabric.
//
// The coordinator (libmsafd_live.so, loaded into HF's process) owns two IPC
// device buffers (one input tile, one output tile) and a shared control block.
// N unit processes (one per transformer layer, MPS-capped) open those buffers and
// serve their layer's FFN on request. Because a forward pass fires layers
// sequentially, only one layer is "active" at a time, so a single in/out buffer
// pair is shared by all units and the control block just names the active layer.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>

namespace msafd {

static const int kLiveMaxUnits = 64;

// Cross-process control block (mmap'd file). Coordinator bumps `epoch` after
// setting `active_layer`+`T`; the matching unit runs its FFN and writes `epoch`
// into `done_epoch`. `ready[L]` is each unit's attach handshake; `stop` tears down.
struct LiveControl {
    volatile int active_layer;   // layer whose unit should run now (-1 = idle)
    volatile int T;              // token count for the current dispatch
    volatile int epoch;          // bumped by coordinator per dispatch
    volatile int done_epoch;     // set by the active unit when finished
    volatile int stop;           // teardown flag
    volatile int ready[kLiveMaxUnits];
    int n_layers;
};

#define LIVE_CUDA_CHECK(expr)                                                   \
    do {                                                                       \
        cudaError_t _e = (expr);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "[live] %s failed at %s:%d: %s\n", #expr,    \
                         __FILE__, __LINE__, cudaGetErrorString(_e));         \
            std::abort();                                                     \
        }                                                                    \
    } while (0)

static inline LiveControl* live_map_control(const char* path, bool create,
                                            int n_layers) {
    int flags = create ? (O_CREAT | O_RDWR | O_TRUNC) : O_RDWR;
    int fd = open(path, flags, 0666);
    if (fd < 0) { std::perror("[live] open ctrl"); std::exit(1); }
    if (create) {
        if (ftruncate(fd, sizeof(LiveControl)) != 0) { std::perror("ftruncate"); std::exit(1); }
    } else {
        struct stat st;
        for (int t = 0;; ++t) {
            if (fstat(fd, &st) == 0 && st.st_size >= (off_t)sizeof(LiveControl)) break;
            if (t > 5000) { std::fprintf(stderr, "[live] ctrl size timeout\n"); std::exit(1); }
            usleep(20000);
        }
    }
    void* p = mmap(nullptr, sizeof(LiveControl), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { std::perror("[live] mmap"); std::exit(1); }
    close(fd);
    LiveControl* c = static_cast<LiveControl*>(p);
    if (create) { std::memset(p, 0, sizeof(LiveControl)); c->n_layers = n_layers;
                  c->active_layer = -1; }
    return c;
}

static inline void live_write_handle(const char* path, const void* data, size_t n) {
    char tmp[1200]; std::snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE* f = std::fopen(tmp, "wb");
    if (!f) { std::perror("[live] write handle"); std::exit(1); }
    std::fwrite(data, n, 1, f); std::fclose(f);
    std::rename(tmp, path);
}

static inline bool live_read_handle(const char* path, void* data, size_t n) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    size_t r = std::fread(data, n, 1, f);
    std::fclose(f);
    return r == 1;
}

}  // namespace msafd
