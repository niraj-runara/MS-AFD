#pragma once
// Minimal fail-fast error checking for CUDA runtime + cuBLAS.
// MS-AFD is deterministic-or-dead: any API error aborts the slice immediately
// rather than limping on with corrupt state.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                         cudaGetErrorName(err__), __FILE__, __LINE__,          \
                         cudaGetErrorString(err__));                           \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t st__ = (call);                                          \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st__,      \
                         __FILE__, __LINE__);                                  \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
