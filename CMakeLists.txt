cmake_minimum_required(VERSION 3.20)

set(HUNTER_PACKAGES abseil)
include(FetchContent)
FetchContent_Declare(SetupHunter GIT_REPOSITORY https://github.com/cpp-pm/gate)
FetchContent_MakeAvailable(SetupHunter)

set(CMAKE_CUDA_COMPILER /home2/sunjianqi/llvm-project/build/bin/clang++)

project(networks LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_package(absl CONFIG REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_ARCHITECTURES OFF)
string(APPEND CMAKE_CUDA_FLAGS "--cuda-gpu-arch=sm_80 -O3 -Xclang -fcuda-allow-variadic-functions -fcuda-short-ptr  -mllvm --nvptx-short-ptr")
#string(APPEND CMAKE_CUDA_FLAGS "--keep -v")

add_executable(naive_lstm naive_lstm.cu)
target_link_libraries(naive_lstm CUDA::cuda_driver absl::base)

add_executable(wavefront_lstm wavefront_lstm.cu)
target_link_libraries(wavefront_lstm CUDA::cuda_driver absl::base)

add_executable(seq2seq seq2seq.cu)
target_link_libraries(seq2seq CUDA::cuda_driver absl::base)

add_executable(bert bert.cu)
target_link_libraries(bert CUDA::cuda_driver absl::base)
