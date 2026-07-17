# MS-AFD — M0 build
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

# M0 single-slice executable.
add_executable(slice
    src/slice.cu
)
target_include_directories(slice PRIVATE src)
target_link_libraries(slice PRIVATE CUDA::cublas CUDA::cudart)

# NCCL is linked from M2 onward (spike S1 tests it earlier).
