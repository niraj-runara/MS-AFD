# MS-AFD build
cmake_minimum_required(VERSION 3.24)
project(ms_afd LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# A100 (Ampere) = 80. 90 = Hopper (H100), 100 = Blackwell (B200).
set(CMAKE_CUDA_ARCHITECTURES 80)

# M0 single-slice executable.
add_executable(slice
    src/slice.cu
    src/ffn.cu
)
set_target_properties(slice PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# cuBLAS for the FFN GEMMs.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(slice PRIVATE CUDA::cublas)

# --- NCCL (for Spike S1 / M2 fabric) -------------------------------------
# Optional: the s1 spike only builds if NCCL is found, so the M0/M1 `slice`
# target still builds on a box without NCCL.
find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS
        $ENV{NCCL_ROOT}/include
        $ENV{CUDA_HOME}/include
        /usr/include
        /usr/local/cuda/include
        /usr/local/cuda/targets/x86_64-linux/include)
find_library(NCCL_LIB nccl
    HINTS
        $ENV{NCCL_ROOT}/lib
        $ENV{CUDA_HOME}/lib64
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/cuda/targets/x86_64-linux/lib)

if(NCCL_INCLUDE_DIR AND NCCL_LIB)
    message(STATUS "NCCL found (${NCCL_LIB}) — building s1_graph_nccl spike")
    add_executable(s1_graph_nccl spikes/s1_graph_nccl.cu)
    target_include_directories(s1_graph_nccl PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_libraries(s1_graph_nccl PRIVATE CUDA::cudart ${NCCL_LIB})
else()
    message(WARNING "NCCL not found — skipping s1_graph_nccl (needed for S1/M2). "
                    "Set NCCL_ROOT if it lives in a nonstandard path.")
endif()
