# MS-AFD build
cmake_minimum_required(VERSION 3.24)
project(ms_afd LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Target GPU: A100 = Ampere = sm_80. (90 = Hopper, 100 = Blackwell.)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(CUDAToolkit REQUIRED)

# M0/M1 single-slice executable (header-only ffn.h / arena.h).
add_executable(slice src/slice.cu)
target_include_directories(slice PRIVATE src)
target_link_libraries(slice PRIVATE CUDA::cublas CUDA::cudart)

# --- NCCL (M2 dense fabric) ----------------------------------------------
# Optional so the M0/M1 `slice` target still builds on a box without NCCL.
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
if(NOT NCCL_LIB)
    # Some distros ship only the versioned soname; the linker wants libnccl.so.
    set(NCCL_LIB /usr/lib/x86_64-linux-gnu/libnccl.so.2)
endif()

if(NCCL_INCLUDE_DIR AND NCCL_LIB)
    message(STATUS "NCCL found (${NCCL_LIB}) — building m2_dense")
    # M2 (dense): A-side scatter -> per-GPU NCCL hub -> IPC fan-out to MPS units.
    add_executable(m2_dense src/m2_dense.cu)
    target_include_directories(m2_dense PRIVATE src ${NCCL_INCLUDE_DIR})
    target_link_libraries(m2_dense PRIVATE CUDA::cublas CUDA::cudart ${NCCL_LIB})
else()
    message(WARNING "NCCL not found — skipping m2_dense (M2). "
                    "Set NCCL_ROOT if it lives in a nonstandard path.")
endif()
