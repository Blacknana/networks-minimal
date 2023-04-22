// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_252
// Description:	Constant
// Input:
// Output:
//	- name: Constant_252_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_252(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_252_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_252_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2479
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2479_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2479(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2479_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2479_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2389
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2389_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2389(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2389_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2389_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_68
// Description:	Constant
// Input:
// Output:
//	- name: Constant_68_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_68(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_68_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_68_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2734
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2734_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2734(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2734_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2734_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2572
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2572_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2572(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2572_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2572_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2617
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2617_0	type: float	shape: Shape{128, 768, 1, 1}
void Constant_float_cuda_Constant_2617(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2617_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2617_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[393216];
  bin_file.read(tmp_mem, 393216);
  cudaMemcpyAsync(output0, tmp_mem, 393216, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2894
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2894_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2894(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2894_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2894_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2035
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2035_0	type: float	shape: Shape{32, 96, 1, 1}
void Constant_float_cuda_Constant_2035(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2035_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2035_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12288];
  bin_file.read(tmp_mem, 12288);
  cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_265
// Description:	Constant
// Input:
// Output:
//	- name: Constant_265_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_265(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_265_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_265_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2824
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2824_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2824(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2824_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2824_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2748
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2748_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2748(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2748_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2748_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1003_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2323_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_1004_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2326_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_1012_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Convolution_1014_0	type: float	shape: Shape{1, 64, 16,
// 16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1012<<<dim3(1, 16, 4), dim3(8,
// 1, 16), 0, 0>>>(DepthwiseConv2dNative_1003_0, Constant_2323_0,
// Convolution_1012_0);
// Convolution_float_float_float_cuda_Convolution_1014<<<dim3(1, 16, 4), dim3(8,
// 1, 16), 0, 0>>>(DepthwiseConv2dNative_1004_0, Constant_2326_0,
// Convolution_1014_0); Deduped function map: <src_function_name :
// deduped_function_name> Convolution_float_float_float_cuda_Convolution_1014 :
// Convolution_float_float_float_cuda_Convolution_1012

// Node name:	Convolution_1012
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1003_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2323_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_1012_0	type: float	shape: Shape{1, 64, 16,
// 16}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1012_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(8, 1, 16);
  const dim3 gridDim(1, 16, 4);
  const dim3 threadIdx(thread_id % 8, 0, thread_id / 8);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 16, block_id / 16);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 1024);
  {
    float *compute = output0;
    {
      float compute_local[2];

      compute_local[0] = 0.000000e+00f;
      compute_local[1] = 0.000000e+00f;
      pad_temp_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input0[(((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                  (((int)threadIdx.x) * 2))];
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      input1_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input1[(((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                  (((int)threadIdx.x) * 2))];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  4096)];
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  4097)];
      input1_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  16)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  17)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  8192)];
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  8193)];
      input1_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  32)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  33)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  12288)];
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  12289)];
      input1_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  48)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  49)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                (((int)blockIdx.y) * 16)) +
               (((int)threadIdx.x) * 2))] = compute_local[0];
      compute[(((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                 (((int)blockIdx.y) * 16)) +
                (((int)threadIdx.x) * 2)) +
               1)] = compute_local[1];
    }
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_72(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {

  __shared__ char shared_buffer[2048];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_1012_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_1012_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_72_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_72<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2781_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1012_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Slice_968_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2881_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1014_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: BatchNormInference_948_0	type: float	shape: Shape{1,
// 64, 16, 16}
// Output:
//	- name: Add_1021_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_1022_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_30<<<dim3(32, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1012_0, Constant_2781_0, Slice_968_0,
// Add_1021_0); FusedKernel_float_float_float_float_cuda_Add_Add_31<<<dim3(32,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1014_0, Constant_2881_0,
// BatchNormInference_948_0, Add_1022_0); Deduped function map:
// <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_31 :
// FusedKernel_float_float_float_float_cuda_Add_Add_30

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1012_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2781_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Slice_968_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1021_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2325<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1012_0, Constant_2781_0, BatchNormInference_1018_0);
// Add_float_float_float_cuda_Add_1021<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1018_0, Slice_968_0, Add_1021_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Add_30_block_kernel(
    float *input0, float *input1, float *input2, float *output0, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(32, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = add(temp0, input2[tid]);
  output0[tid] = temp1;
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_73(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    FusedKernel_float_float_float_float_cuda_Add_Add_30_block_kernel(
        input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    FusedKernel_float_float_float_float_cuda_Add_Add_30_block_kernel(
        input4, input3, input5, output1, threadIdx.x, blockIdx.x - 32, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_73_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_73<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1229_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2452_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3104_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1230_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2455_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3106_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1254_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2464_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_1252_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2458_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_1253_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2461_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Relu_1250_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1251_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1262_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Convolution_1258_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Convolution_1260_0	type: float	shape: Shape{1, 64, 16,
// 16}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3103<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_1229_0,
// Constant_2452_0, Constant_3104_0, Relu_1250_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3105<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_1230_0,
// Constant_2455_0, Constant_3106_0, Relu_1251_0);
// Convolution_float_float_float_cuda_Convolution_1262<<<dim3(1, 16, 4), dim3(8,
// 1, 16), 0, 0>>>(DepthwiseConv2dNative_1254_0, Constant_2464_0,
// Convolution_1262_0);
// Convolution_float_float_float_cuda_Convolution_1258<<<dim3(1, 16, 4), dim3(8,
// 1, 16), 0, 0>>>(DepthwiseConv2dNative_1252_0, Constant_2458_0,
// Convolution_1258_0);
// Convolution_float_float_float_cuda_Convolution_1260<<<dim3(1, 16, 4), dim3(8,
// 1, 16), 0, 0>>>(DepthwiseConv2dNative_1253_0, Constant_2461_0,
// Convolution_1260_0); Deduped function map: <src_function_name :
// deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3105 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3103
// Convolution_float_float_float_cuda_Convolution_1258 :
// Convolution_float_float_float_cuda_Convolution_1262
// Convolution_float_float_float_cuda_Convolution_1260 :
// Convolution_float_float_float_cuda_Convolution_1262

// Node name:	Matched_Pattern_3103
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1229_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2452_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3104_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1250_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3103_block_kernel(
    float *input0, float *input1, float *input2, float *output0, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(8, 1, 16);
  const dim3 gridDim(1, 16, 4);
  const dim3 threadIdx(thread_id % 8, 0, thread_id / 8);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 16, block_id / 16);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 1024);
  {
    float *compute = output0;
    {
      float compute1[2];

      compute1[0] = 0.000000e+00f;
      compute1[1] = 0.000000e+00f;
      pad_temp_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input0[(((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                  (((int)threadIdx.x) * 2))];
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      input1_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input1[(((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                  (((int)threadIdx.x) * 2))];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      __syncthreads();
      compute1[0] = (compute1[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                                    input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                          input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  4096)];
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  4097)];
      input1_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  16)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  17)];
      __syncthreads();
      compute1[0] = (compute1[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                                    input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                          input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  8192)];
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  8193)];
      input1_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  32)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  33)];
      __syncthreads();
      compute1[0] = (compute1[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                                    input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                          input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  12288)];
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  12289)];
      input1_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  48)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  49)];
      __syncthreads();
      compute1[0] = (compute1[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                                    input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                          input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute1[1] =
          (compute1[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
                          input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                (((int)blockIdx.y) * 16)) +
               (((int)threadIdx.x) * 2))] =
          max((compute1[0] +
               input2[((((int)blockIdx.z) * 16) + ((int)threadIdx.z))]),
              0.000000e+00f);
      compute[(((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                 (((int)blockIdx.y) * 16)) +
                (((int)threadIdx.x) * 2)) +
               1)] =
          max((compute1[1] +
               input2[((((int)blockIdx.z) * 16) + ((int)threadIdx.z))]),
              0.000000e+00f);
    }
  }
}
// Node name:	Convolution_1262
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1254_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2464_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_1262_0	type: float	shape: Shape{1, 64, 16,
// 16}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1262_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(8, 1, 16);
  const dim3 gridDim(1, 16, 4);
  const dim3 threadIdx(thread_id % 8, 0, thread_id / 8);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 16, block_id / 16);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 1024);
  {
    float *compute = output0;
    {
      float compute_local[2];

      compute_local[0] = 0.000000e+00f;
      compute_local[1] = 0.000000e+00f;
      pad_temp_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input0[(((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                  (((int)threadIdx.x) * 2))];
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      input1_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input1[(((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                  (((int)threadIdx.x) * 2))];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  4096)];
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  4097)];
      input1_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  16)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  17)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  8192)];
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  8193)];
      input1_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  32)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  33)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  12288)];
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  12289)];
      input1_shared[((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  48)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                   (((int)threadIdx.x) * 2)) +
                  49)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                (((int)blockIdx.y) * 16)) +
               (((int)threadIdx.x) * 2))] = compute_local[0];
      compute[(((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                 (((int)blockIdx.y) * 16)) +
                (((int)threadIdx.x) * 2)) +
               1)] = compute_local[1];
    }
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_106(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {

  __shared__ char shared_buffer[2048];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3103_block_kernel(
        input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3103_block_kernel(
        input3, input4, input5, output1, threadIdx.x, blockIdx.x - 64,
        shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    Convolution_float_float_float_cuda_Convolution_1262_block_kernel(
        input6, input7, output2, threadIdx.x, blockIdx.x - 128, shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    Convolution_float_float_float_cuda_Convolution_1262_block_kernel(
        input8, input9, output3, threadIdx.x, blockIdx.x - 192, shared_buffer);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319) {
    Convolution_float_float_float_cuda_Convolution_1262_block_kernel(
        input10, input11, output4, threadIdx.x, blockIdx.x - 256,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_106_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_106<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, input11, output0, output1, output2, output3, output4);
}
