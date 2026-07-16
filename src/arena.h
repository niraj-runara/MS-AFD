#pragma once
// M0 · Step 2 — Static 1 GB HBM arena + fixed-offset allocator.
//
// Rules the implementation must enforce:
//   - Exactly ONE cudaMalloc (the 1 GB arena) at startup. Never again.
//   - Tensors are handed out at FIXED offsets (input / output / weights / workspace).
//   - Any request that overflows its region hard-crashes the slice (boundary by
//     construction, not by policy).
//
// Boilerplate only — no implementation yet.

#include <cstddef>

namespace msafd {

constexpr std::size_t kArenaBytes = 1ull << 30;  // 1 GB

class Arena {
public:
    // Allocate the single 1 GB device buffer. TODO: cudaMalloc, store base ptr.
    explicit Arena(std::size_t bytes = kArenaBytes);
    ~Arena();

    // Hand out a region at the current fixed offset. TODO: bump offset, bounds-check,
    // abort() on overflow. Returns device pointer.
    void* alloc(std::size_t bytes, const char* tag);

    std::size_t used() const;      // TODO
    std::size_t capacity() const;  // TODO

private:
    void*       base_   = nullptr;  // single device allocation
    std::size_t offset_ = 0;
    std::size_t cap_    = 0;
};

}  // namespace msafd
