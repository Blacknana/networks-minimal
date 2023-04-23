// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
#define MIN(a, b) ((a) > (b) ? (b) : (a))
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }

#if CUDA_VERSION < 9000
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate)                                      \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

__forceinline__ __device__ float
CudaShuffleDownSync(unsigned mask, float val, int delta, int width = 32) {
#if CUDA_VERSION < 9000
  return __shfl_down(val, delta, width);
#else
  return __shfl_down_sync(mask, val, delta, width);
#endif
}

__device__ static float reduceMax(float val, int tid, int blockSize,
                                  float *shm) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val = max(val, CudaShuffleDownSync(mask, val, 16));
  val = max(val, CudaShuffleDownSync(mask, val, 8));
  val = max(val, CudaShuffleDownSync(mask, val, 4));
  val = max(val, CudaShuffleDownSync(mask, val, 2));
  val = max(val, CudaShuffleDownSync(mask, val, 1));

  if (tid < warpSize)
    shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0)
    shm[tid / warpSize] = val;
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];

    val = max(val, CudaShuffleDownSync(mask, val, 16));
    val = max(val, CudaShuffleDownSync(mask, val, 8));
    val = max(val, CudaShuffleDownSync(mask, val, 4));
    val = max(val, CudaShuffleDownSync(mask, val, 2));
    val = max(val, CudaShuffleDownSync(mask, val, 1));
  }

  return val;
}

__device__ static float reduceSum(float val, int tid, int blockSize,
                                  float *shm) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val += CudaShuffleDownSync(mask, val, 16);
  val += CudaShuffleDownSync(mask, val, 8);
  val += CudaShuffleDownSync(mask, val, 4);
  val += CudaShuffleDownSync(mask, val, 2);
  val += CudaShuffleDownSync(mask, val, 1);

  if (tid < warpSize)
    shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0)
    shm[tid / warpSize] = val;

  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];

    val += CudaShuffleDownSync(mask, val, 16);
    val += CudaShuffleDownSync(mask, val, 8);
    val += CudaShuffleDownSync(mask, val, 4);
    val += CudaShuffleDownSync(mask, val, 2);
    val += CudaShuffleDownSync(mask, val, 1);
  }

  return val;
}
// Node name:	Constant_130
// Description:	Constant
// Input:
// Output:
//	- name: Constant_130_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_130(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_130_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_130_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2677
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2677_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2677(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2677_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2677_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3164
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3164_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3164(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3164_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3164_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2245
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2245_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2245(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2245_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2245_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_232
// Description:	Constant
// Input:
// Output:
//	- name: Constant_232_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_232(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_232_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_232_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2494
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2494_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2494(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2494_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2494_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3080
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3080_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3080(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3080_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3080_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2602
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2602_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2602(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2602_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2602_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2425
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2425_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2425(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2425_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2425_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2098
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2098_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2098(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2098_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2098_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_87
// Description:	Constant
// Input:
// Output:
//	- name: Constant_87_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_87(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_87_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_87_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2089
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2089_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2089(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2089_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2089_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3134
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3134_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3134(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3134_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3134_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2332
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2332_0	type: float	shape: Shape{64, 384, 1, 1}
void Constant_float_cuda_Constant_2332(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2332_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2332_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[98304];
  bin_file.read(tmp_mem, 98304);
  cudaMemcpyAsync(output0, tmp_mem, 98304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Sum_1757
// Description:	Sum
// Input:
//	- name: Relu_1756_0	type: float	shape: Shape{1, 768, 8, 8}
// Output:
//	- name: Sum_1757_0	type: float	shape: Shape{1, 768}
extern "C" __launch_bounds__(64) __global__
    void Sum_float_float_cuda_Sum_1757(float *input0, float *output0) {

  int width = 64;
  int block_size = 64;
  const int warp_size = 32;
  __shared__ float shm[warp_size];

  int thread_idx = threadIdx.x;
  int block_idx = blockIdx.x;
  int data_idx_offset = block_idx * width;

  float val = 0.0;
  for (int tidx = thread_idx; tidx < width; tidx += block_size) {
    int data_idx = tidx + data_idx_offset;
    val += input0[data_idx];
  }
  val = reduceSum(val, thread_idx, block_size, shm);
  if (thread_idx == 0)
    output0[block_idx] = val;
}
extern void Sum_float_float_cuda_Sum_1757_Call(const dim3 &grids,
                                               const dim3 &blocks, unsigned mem,
                                               cudaStream_t stream,
                                               float *input0, float *output0) {
  Sum_float_float_cuda_Sum_1757<<<grids, blocks, mem, stream>>>(input0,
                                                                output0);
}
