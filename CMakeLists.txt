# MS-AFD — M0 build (boilerplate)
cmake_minimum_required(VERSION 3.24)
project(ms_afd LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# TODO: set to your GPU. 90 = Hopper (H100), 100 = Blackwell (B200).
set(CMAKE_CUDA_ARCHITECTURES 90)

# M0 single-slice executable.
add_executable(slice
    src/slice.cu
)

# TODO: link cuBLAS (and NCCL once S1/M2 needs it).
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(slice PRIVATE CUDA::cublas)
