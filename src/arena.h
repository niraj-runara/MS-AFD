#pragma once
// M0 · Step 2 — Static HBM arena + fixed-offset allocator.
//
// Rules the implementation enforces:
//   - Exactly ONE cudaMalloc (the arena) at startup. Never again.
//   - Tensors are handed out at FIXED, monotonically bumped offsets
//     (input / output / weights / workspace).
//   - Any request that overflows the arena hard-crashes the slice (boundary by
//     construction, not by policy).
//
// Header-only: all methods are inline so multiple .cu TUs can include it.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "check.h"

namespace msafd {

// Per-slice static arena. A Qwen3-30B-A3B expert is only ~9 MB of weights;
// with activations + cuBLAS workspace the whole slice fits well under 32 MB.
// 64 MB is generous headroom while keeping HBM from being what caps slice count
// in M1 (we want MPS SM% to be the real limiter, not memory).
constexpr std::size_t kArenaBytes = 64ull << 20;  // 64 MB

class Arena {
public:
    // The single device allocation for this slice.
    explicit Arena(std::size_t bytes = kArenaBytes) : cap_(bytes) {
        CUDA_CHECK(cudaMalloc(&base_, cap_));
    }

    ~Arena() {
        if (base_) cudaFree(base_);  // best-effort; process teardown anyway
    }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Hand out a 256B-aligned region at the current offset. Aborts on overflow.
    void* alloc(std::size_t bytes, const char* tag) {
        constexpr std::size_t kAlign = 256;
        std::size_t start = (offset_ + kAlign - 1) & ~(kAlign - 1);
        if (start + bytes > cap_) {
            std::fprintf(stderr,
                         "MS-AFD arena overflow on '%s': need %zu B at offset "
                         "%zu, cap %zu B\n",
                         tag, bytes, start, cap_);
            std::abort();
        }
        void* p = static_cast<char*>(base_) + start;
        offset_ = start + bytes;
        return p;
    }

    std::size_t used() const { return offset_; }
    std::size_t capacity() const { return cap_; }

private:
    void*       base_   = nullptr;  // single device allocation
    std::size_t offset_ = 0;
    std::size_t cap_    = 0;
};

}  // namespace msafd
