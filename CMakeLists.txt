cmake_minimum_required(VERSION 3.20)

set(HUNTER_PACKAGES abseil GTest)
include(FetchContent)
FetchContent_Declare(SetupHunter GIT_REPOSITORY https://github.com/cpp-pm/gate)
FetchContent_MakeAvailable(SetupHunter)

option(USE_CLANG OFF)
if(USE_CLANG)
  set(CMAKE_CUDA_COMPILER /llvm-project/build/bin/clang++)
endif()

project(networks LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_package(absl CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_STANDARD 20)
if(USE_CLANG)
  set(CMAKE_CUDA_ARCHITECTURES OFF)
  string(APPEND CMAKE_CUDA_FLAGS "--cuda-gpu-arch=sm_80 -O3 -Xclang -fcuda-allow-variadic-functions -fcuda-short-ptr  -mllvm --nvptx-short-ptr")
else()
  set(CMAKE_CUDA_ARCHITECTURES 80)
  string(APPEND CMAKE_CUDA_FLAGS "--keep -v")
endif()

add_subdirectory(lstm)
add_subdirectory(seq2seq)
add_subdirectory(bert)