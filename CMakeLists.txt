# MS-AFD — M0 build (boilerplate)
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

# cuBLAS for the FFN GEMMs (NCCL added once S1/M2 needs it).
find_package(CUDAToolkit REQUIRED)
target_link_libraries(slice PRIVATE CUDA::cublas)
