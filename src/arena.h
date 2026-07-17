#pragma once
// M0 · Step 2 — Static 1 GB HBM arena + fixed-offset allocator.
//
// Rules enforced here:
//   - Exactly ONE cudaMalloc (the 1 GB arena) at startup. Never again.
//   - Tensors are handed out at FIXED offsets (weights / activations / workspace).
//   - Any request that overflows the arena hard-crashes the slice (boundary by
//     construction, not by policy).
//
// Header-only so the single-TU M0 build (only slice.cu) stays trivial.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace msafd {

constexpr std::size_t kArenaBytes = 1ull << 30;  // 1 GB

#define MSAFD_CUDA_CHECK(expr)                                                  \
    do {                                                                       \
        cudaError_t _e = (expr);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "[cuda] %s failed at %s:%d: %s\n", #expr,    \
                         __FILE__, __LINE__, cudaGetErrorString(_e));         \
            std::abort();                                                     \
        }                                                                    \
    } while (0)

class Arena {
public:
    explicit Arena(std::size_t bytes = kArenaBytes) : cap_(bytes) {
        MSAFD_CUDA_CHECK(cudaMalloc(&base_, bytes));
        std::printf("[arena] reserved %zu bytes (single cudaMalloc)\n", bytes);
    }
    ~Arena() {
        if (base_) cudaFree(base_);
    }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Hand out a 256-byte-aligned region at the current fixed offset.
    // Bounds-checked: overflow aborts the slice (hard crash, by construction).
    void* alloc(std::size_t bytes, const char* tag) {
        constexpr std::size_t kAlign = 256;
        std::size_t start = (offset_ + kAlign - 1) & ~(kAlign - 1);
        if (start + bytes > cap_) {
            std::fprintf(stderr,
                         "[arena] OVERFLOW: '%s' wants %zu B at offset %zu, "
                         "cap %zu\n",
                         tag, bytes, start, cap_);
            std::abort();
        }
        void* p = static_cast<char*>(base_) + start;
        offset_ = start + bytes;
        std::printf("[arena] +%-12zu B  %-14s  (used %zu / %zu)\n", bytes, tag,
                    offset_, cap_);
        return p;
    }

    std::size_t used() const { return offset_; }
    std::size_t capacity() const { return cap_; }

private:
    void*       base_   = nullptr;  // the single device allocation
    std::size_t offset_ = 0;
    std::size_t cap_    = 0;
};

}  // namespace msafd
