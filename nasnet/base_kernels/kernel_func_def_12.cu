// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_3044
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3044_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_3044(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3044_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3044_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_96
// Description:	Constant
// Input:
// Output:
//	- name: Constant_96_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_96(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_96_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_96_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3042
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3042_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3042(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3042_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3042_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2800
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2800_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2800(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2800_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2800_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_312
// Description:	Constant
// Input:
// Output:
//	- name: Constant_312_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_312(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_312_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_312_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_1338
// Description:	Constant
// Input:
// Output:
//	- name: Constant_1338_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1338(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_1338_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_1338_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4];
  bin_file.read(tmp_mem, 4);
  cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2026
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2026_0	type: float	shape: Shape{32, 96, 1, 1}
void Constant_float_cuda_Constant_2026(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2026_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2026_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12288];
  bin_file.read(tmp_mem, 12288);
  cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2014
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2014_0	type: float	shape: Shape{32, 96, 1, 1}
void Constant_float_cuda_Constant_2014(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2014_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2014_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12288];
  bin_file.read(tmp_mem, 12288);
  cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2128
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2128_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2128(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2128_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2128_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_251
// Description:	Constant
// Input:
// Output:
//	- name: Constant_251_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_251(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_251_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_251_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[3200];
  bin_file.read(tmp_mem, 3200);
  cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1087_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2365_0	type: float	shape: Shape{64, 384, 1, 1}
//	- name: Constant_2368_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1089_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Convolution_1091_0	type: float	shape: Shape{1, 64, 16,
// 16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1089<<<dim3(1, 16, 4),
// dim3(16, 1, 16), 0, 0>>>(Relu_1087_0, Constant_2365_0, Convolution_1089_0);
// Convolution_float_float_float_cuda_Convolution_1091<<<dim3(1, 16, 4),
// dim3(16, 1, 16), 0, 0>>>(Relu_1087_0, Constant_2368_0, Convolution_1091_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1091 :
// Convolution_float_float_float_cuda_Convolution_1089

// Node name:	Convolution_1089
// Description:	Convolution
// Input:
//	- name: Relu_1087_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2365_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1089_0	type: float	shape: Shape{1, 64, 16,
// 16}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1089_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(16, 1, 16);
  const dim3 gridDim(1, 16, 4);
  const dim3 threadIdx(thread_id % 16, 0, thread_id / 16);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 16, block_id / 16);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 3072);
  {
    float *compute = output0;
    {
      float compute_local[1];

      compute_local[0] = 0.000000e+00f;
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         (((((int)threadIdx.x) * 3) / 16) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       ((((int)threadIdx.x) * 3) & 15))]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       (((((int)threadIdx.x) * 3) + 1) & 15))]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       (((((int)threadIdx.x) * 3) + 2) & 15))]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[(((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                  (((int)threadIdx.x) * 3))];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  1)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  2)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       12288)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       12288)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       12288)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  48)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  49)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  50)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       24576)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       24576)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       24576)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  96)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  97)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  98)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       36864)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       36864)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       36864)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  144)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  145)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  146)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       49152)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  192)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  193)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  194)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       61440)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       61440)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       61440)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  240)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  241)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  242)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       73728)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       73728)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       73728)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  288)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  289)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  290)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       86016)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       86016)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       86016)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  336)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  337)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  338)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                (((int)blockIdx.y) * 16)) +
               ((int)threadIdx.x))] = compute_local[0];
    }
  }
}
__device__ __forceinline__ void
fuse2_Convolution_float_float_float_cuda_Convolution_1089_block_kernel(
    float *input0, float *input1, float *input2, float *output0, float *output1,
    int thread_id, int block_id, char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(16, 1, 16);
  const dim3 gridDim(1, 16, 4);
  const dim3 threadIdx(thread_id % 16, 0, thread_id / 16);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 16, block_id / 16);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 3072);
  float *input1_shared1 = (float *)(shared_buffer + 6144);
  {
    float *compute = output0;
    {
      float compute_local[1];
      float compute_local1[1];

      compute_local[0] = 0.000000e+00f;
      compute_local1[0] = 0.000000e+00f;
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         (((((int)threadIdx.x) * 3) / 16) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       ((((int)threadIdx.x) * 3) & 15))]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       (((((int)threadIdx.x) * 3) + 1) & 15))]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       (((((int)threadIdx.x) * 3) + 2) & 15))]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[(((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                  (((int)threadIdx.x) * 3))];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  1)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  2)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[(((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                  (((int)threadIdx.x) * 3))];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  1)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  2)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       12288)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       12288)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       12288)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  48)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  49)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  50)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  48)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  49)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  50)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       24576)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       24576)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       24576)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  96)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  97)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  98)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  96)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  97)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  98)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       36864)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       36864)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       36864)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  144)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  145)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  146)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  144)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  145)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  146)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       49152)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  192)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  193)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  194)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  192)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  193)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  194)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       61440)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       61440)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       61440)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  240)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  241)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  242)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  240)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  241)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  242)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       73728)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       73728)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       73728)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  288)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  289)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  290)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  288)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  289)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  290)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       86016)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       86016)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       86016)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  336)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  337)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  338)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  336)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  337)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  338)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      compute[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                (((int)blockIdx.y) * 16)) +
               ((int)threadIdx.x))] = compute_local[0];
      output1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                (((int)blockIdx.y) * 16)) +
               ((int)threadIdx.x))] = compute_local1[0];
    }
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_83(
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {

  __shared__ char shared_buffer[9216];
  // __shared__ char shared_buffer[6144];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    fuse2_Convolution_float_float_float_cuda_Convolution_1089_block_kernel(
        input0, input1, input2, output0, output1, threadIdx.x, blockIdx.x - 0,
        shared_buffer);
    // Convolution_float_float_float_cuda_Convolution_1089_block_kernel(input0,
    // input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  }
  // else if((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
  // {
  //     Convolution_float_float_float_cuda_Convolution_1089_block_kernel(input0,
  //     input2, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  // }
}

extern void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_83_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_83<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}

// Node name:	 BlockFusion
// Input:
//	- name: Relu_1024_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2329_0	type: float	shape: Shape{64, 384, 1, 1}
//	- name: Constant_2332_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1026_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Convolution_1028_0	type: float	shape: Shape{1, 64, 16,
// 16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1026<<<dim3(1, 16, 4),
// dim3(16, 1, 16), 0, 0>>>(Relu_1024_0, Constant_2329_0, Convolution_1026_0);
// Convolution_float_float_float_cuda_Convolution_1028<<<dim3(1, 16, 4),
// dim3(16, 1, 16), 0, 0>>>(Relu_1024_0, Constant_2332_0, Convolution_1028_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1028 :
// Convolution_float_float_float_cuda_Convolution_1026

// Node name:	Convolution_1026
// Description:	Convolution
// Input:
//	- name: Relu_1024_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2329_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1026_0	type: float	shape: Shape{1, 64, 16,
// 16}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1026_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(16, 1, 16);
  const dim3 gridDim(1, 16, 4);
  const dim3 threadIdx(thread_id % 16, 0, thread_id / 16);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 16, block_id / 16);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 3072);
  {
    float *compute = output0;
    {
      float compute_local[1];

      compute_local[0] = 0.000000e+00f;
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         (((((int)threadIdx.x) * 3) / 16) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       ((((int)threadIdx.x) * 3) & 15))]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       (((((int)threadIdx.x) * 3) + 1) & 15))]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       (((((int)threadIdx.x) * 3) + 2) & 15))]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[(((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                  (((int)threadIdx.x) * 3))];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  1)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  2)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       12288)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       12288)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       12288)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  48)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  49)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  50)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       24576)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       24576)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       24576)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  96)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  97)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  98)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       36864)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       36864)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       36864)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  144)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  145)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  146)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       49152)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  192)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  193)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  194)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       61440)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       61440)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       61440)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  240)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  241)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  242)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       73728)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       73728)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       73728)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  288)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  289)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  290)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       86016)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       86016)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       86016)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  336)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  337)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  338)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                (((int)blockIdx.y) * 16)) +
               ((int)threadIdx.x))] = compute_local[0];
    }
  }
}
__device__ __forceinline__ void
fuse2_Convolution_float_float_float_cuda_Convolution_1026_block_kernel(
    float *input0, float *input1, float *input2, float *output0, float *output1,
    int thread_id, int block_id, char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(16, 1, 16);
  const dim3 gridDim(1, 16, 4);
  const dim3 threadIdx(thread_id % 16, 0, thread_id / 16);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 16, block_id / 16);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 3072);
  float *input1_shared1 = (float *)(shared_buffer + 6144);
  {
    float *compute = output0;
    {
      float compute_local[1];
      float compute_local1[1];

      compute_local[0] = 0.000000e+00f;
      compute_local1[0] = 0.000000e+00f;
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         (((((int)threadIdx.x) * 3) / 16) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       ((((int)threadIdx.x) * 3) & 15))]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       (((((int)threadIdx.x) * 3) + 1) & 15))]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       (((((int)threadIdx.x) * 3) + 2) & 15))]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[(((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                  (((int)threadIdx.x) * 3))];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  1)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  2)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[(((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                  (((int)threadIdx.x) * 3))];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  1)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  2)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       12288)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       12288)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       12288)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  48)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  49)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  50)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  48)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  49)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  50)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       24576)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       24576)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       24576)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  96)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  97)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  98)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  96)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  97)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  98)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       36864)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       36864)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       36864)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  144)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  145)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  146)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  144)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  145)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  146)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       49152)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  192)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  193)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  194)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  192)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  193)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  194)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       61440)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       61440)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       61440)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  240)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  241)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  242)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  240)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  241)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  242)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       73728)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       73728)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       73728)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  288)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  289)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  290)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  288)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  289)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  290)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       86016)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       86016)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       86016)]);
      input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  336)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  337)];
      input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  338)];
      input1_shared1[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  336)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  337)];
      input1_shared1[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                   (((int)threadIdx.x) * 3)) +
                  338)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 48) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 15)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 17)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 18)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 19)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 20)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 21)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 22)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 23)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 24)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 25)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 26)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 27)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 28)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 29)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 30)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 31)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 32)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 33)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 34)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 35)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 36)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 37)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 38)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 39)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 40)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 41)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 42)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 43)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 44)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 45)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 46)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                           input1_shared[((((int)threadIdx.z) * 48) + 47)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 15)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 272)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 17)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 18)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 304)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 19)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 20)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 336)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 21)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 22)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 368)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 23)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 24)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 400)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 25)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 26)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 432)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 27)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 28)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 464)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 29)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 30)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 496)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 31)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 32)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 528)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 33)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 34)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 560)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 35)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 36)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 592)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 37)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 38)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 624)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 39)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 40)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 656)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 41)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 42)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 688)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 43)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 44)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 720)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 45)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 46)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 752)] *
                            input1_shared1[((((int)threadIdx.z) * 48) + 47)]));
      compute[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                (((int)blockIdx.y) * 16)) +
               ((int)threadIdx.x))] = compute_local[0];
      output1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                (((int)blockIdx.y) * 16)) +
               ((int)threadIdx.x))] = compute_local1[0];
    }
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_74(
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {

  __shared__ char shared_buffer[9216];
  // __shared__ char shared_buffer[6144];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    fuse2_Convolution_float_float_float_cuda_Convolution_1026_block_kernel(
        input0, input1, input2, output0, output1, threadIdx.x, blockIdx.x - 0,
        shared_buffer);
    // Convolution_float_float_float_cuda_Convolution_1026_block_kernel(input0,
    // input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  }
  // else if((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
  // {
  //     Convolution_float_float_float_cuda_Convolution_1026_block_kernel(input0,
  //     input2, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  // }
}

extern void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_74_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_74<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_832_0	type: float	shape: Shape{1, 64, 32, 32}
//	- name: Constant_366_0	type: float	shape: Shape{7, 7, 64, 1}
//	- name: Constant_346_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Constant_348_0	type: float	shape: Shape{7, 7, 64, 1}
//	- name: Convolution_829_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Constant_2740_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: DepthwiseConv2dNative_836_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: DepthwiseConv2dNative_834_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: DepthwiseConv2dNative_835_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: BatchNormInference_831_0	type: float	shape: Shape{1,
// 32, 32, 32}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_836<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_832_0, Constant_366_0,
// DepthwiseConv2dNative_836_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_834<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_832_0, Constant_346_0,
// DepthwiseConv2dNative_834_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_835<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_832_0, Constant_348_0,
// DepthwiseConv2dNative_835_0); Add_float_float_float_cuda_Add_2229<<<dim3(64,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_829_0, Constant_2740_0,
// BatchNormInference_831_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_835 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_836
__device__ __forceinline__ static void
fused2_DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_836_block_kernel(
    float *input0, float *input1, float *input2, float *output0, float *output1,
    int thread_id, int block_id, char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(128, 1, 1);
  const dim3 gridDim(128, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);

  typedef float S;
  float *input = input0;
  float *filter = input1;
  float *output = output0;

  const int in_height = 32;
  const int in_width = 32;
  const int in_depth = 64;
  const int filter_height = 7;
  const int filter_width = 7;
  const int depth_multiplier = 1;
  const int stride = 2;
  const int pad_height = 2;
  const int pad_width = 2;
  const int out_height = 16;
  const int out_width = 16;
  const int out_depth = 64;
  const int num_outputs = 16384;

  for (uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
       thread_id < num_outputs; thread_id += blockDim.x * gridDim.x) {
    // Compute the indexes of this thread in the output.
    //
    // We want coalesced reads so we make sure that each warp reads
    // a contiguous chunk of memory.
    //
    // THIS IS PROBABLY WRONG, we are not doing coalesced reads
    // into the input, because of the depth multiplier division...
    const int out_col = thread_id % out_width;
    const int out_row = (thread_id / out_width) % out_height;
    const int out_channel = (thread_id / out_width / out_height) % out_depth;
    const int batch = thread_id / out_width / out_height / out_depth;

    // Compute the input depth and the index of depth multiplier
    // based off the output depth index that this thread is
    // computing n.
    const int in_channel = out_channel / depth_multiplier;
    const int multiplier = out_channel % depth_multiplier;

    // Data is stored in the following format (let's assume we
    // flatten the height and width into one contiguous dimension
    // called "P".
    //
    // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
    //
    // Each row contains in_depth * in_height * in_width values
    // for each sample in the batch.
    //
    // We can further flatten it into:
    //
    // B1C1P1 B1C1P2 .....
    // B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 .....
    // B2C2P1 B2C2P2 ....
    //
    // where each row is a contiguous array of all of the spatial
    // pixels for a given batch and input depth.  The following
    // loop #pragma unrolls across the filter dimensions for a given thread,
    // indexing into the filter value and the corresponding input
    // patch.
    //
    // We can compute the index into the patch once right here.
    const int input_offset_temp =
        (batch * in_depth + in_channel) * (in_height * in_width);

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    //
    // We perform an additional optimization, where we can determine
    // whether the patch fits within the image indices statically, and
    // avoid boundary checking within the loop.
    const int input_row_start = out_row * stride - pad_height;
    const int input_col_start = out_col * stride - pad_width;
    const int input_row_end = input_row_start + filter_height;
    const int input_col_end = input_col_start + filter_width;

    S sum = static_cast<S>(0);
    S sum2 = static_cast<S>(0);
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < in_height && input_col_end < in_width) {
// Loop that doesn't need to check for boundary conditions.
#pragma unroll
      for (int filter_row = 0; filter_row < filter_height; ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
#pragma unroll
        for (int filter_col = 0; filter_col < filter_width; ++filter_col) {
          const int in_col = input_col_start + filter_col;

          const int input_offset =
              (input_offset_temp) + (in_row * in_width) + in_col;
          const int filter_offset =
              multiplier +
              depth_multiplier *
                  (in_channel + in_depth * (filter_col + filter_offset_temp));
          sum += static_cast<S>(__ldg(input + input_offset)) *
                 static_cast<S>(__ldg(filter + filter_offset));
          sum2 += static_cast<S>(__ldg(input + input_offset)) *
                  static_cast<S>(__ldg(input2 + filter_offset));
        }
      }
    } else {
// Loop that needs to check for boundary conditions.
#pragma unroll
      for (int filter_row = 0; filter_row < filter_height; ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
#pragma unroll
        for (int filter_col = 0; filter_col < filter_width; ++filter_col) {
          const int in_col = input_col_start + filter_col;
          // TODO(vrv): the in_row check can be done outside of this loop;
          // benchmark both methods to determine the better decision.
          if (in_row >= 0 && in_row < in_height && in_col >= 0 &&
              in_col < in_width) {
            const int in_col = input_col_start + filter_col;

            // input_offset_temp indexes into the start of memory
            // where the spatial data starts.
            const int input_offset =
                (input_offset_temp) + (in_row * in_width) + in_col;

            const int filter_offset =
                multiplier +
                depth_multiplier *
                    (in_channel + in_depth * (filter_col + filter_offset_temp));
            sum += static_cast<S>(__ldg(input + input_offset)) *
                   static_cast<S>(__ldg(filter + filter_offset));
            sum2 += static_cast<S>(__ldg(input + input_offset)) *
                    static_cast<S>(__ldg(input2 + filter_offset));
          }
        }
      }
    }

    output[thread_id] = static_cast<S>(sum);
    output1[thread_id] = static_cast<S>(sum2);
  }
}
// Node name:	DepthwiseConv2dNative_836
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_832_0	type: float	shape: Shape{1, 64, 32, 32}
//	- name: Constant_366_0	type: float	shape: Shape{7, 7, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_836_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_836_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(128, 1, 1);
  const dim3 gridDim(128, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);

  typedef float S;
  float *input = input0;
  float *filter = input1;
  float *output = output0;

  const int in_height = 32;
  const int in_width = 32;
  const int in_depth = 64;
  const int filter_height = 7;
  const int filter_width = 7;
  const int depth_multiplier = 1;
  const int stride = 2;
  const int pad_height = 2;
  const int pad_width = 2;
  const int out_height = 16;
  const int out_width = 16;
  const int out_depth = 64;
  const int num_outputs = 16384;

  for (uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
       thread_id < num_outputs; thread_id += blockDim.x * gridDim.x) {
    // Compute the indexes of this thread in the output.
    //
    // We want coalesced reads so we make sure that each warp reads
    // a contiguous chunk of memory.
    //
    // THIS IS PROBABLY WRONG, we are not doing coalesced reads
    // into the input, because of the depth multiplier division...
    const int out_col = thread_id % out_width;
    const int out_row = (thread_id / out_width) % out_height;
    const int out_channel = (thread_id / out_width / out_height) % out_depth;
    const int batch = thread_id / out_width / out_height / out_depth;

    // Compute the input depth and the index of depth multiplier
    // based off the output depth index that this thread is
    // computing n.
    const int in_channel = out_channel / depth_multiplier;
    const int multiplier = out_channel % depth_multiplier;

    // Data is stored in the following format (let's assume we
    // flatten the height and width into one contiguous dimension
    // called "P".
    //
    // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
    //
    // Each row contains in_depth * in_height * in_width values
    // for each sample in the batch.
    //
    // We can further flatten it into:
    //
    // B1C1P1 B1C1P2 .....
    // B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 .....
    // B2C2P1 B2C2P2 ....
    //
    // where each row is a contiguous array of all of the spatial
    // pixels for a given batch and input depth.  The following
    // loop #pragma unrolls across the filter dimensions for a given thread,
    // indexing into the filter value and the corresponding input
    // patch.
    //
    // We can compute the index into the patch once right here.
    const int input_offset_temp =
        (batch * in_depth + in_channel) * (in_height * in_width);

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    //
    // We perform an additional optimization, where we can determine
    // whether the patch fits within the image indices statically, and
    // avoid boundary checking within the loop.
    const int input_row_start = out_row * stride - pad_height;
    const int input_col_start = out_col * stride - pad_width;
    const int input_row_end = input_row_start + filter_height;
    const int input_col_end = input_col_start + filter_width;

    S sum = static_cast<S>(0);
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < in_height && input_col_end < in_width) {
// Loop that doesn't need to check for boundary conditions.
#pragma unroll
      for (int filter_row = 0; filter_row < filter_height; ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
#pragma unroll
        for (int filter_col = 0; filter_col < filter_width; ++filter_col) {
          const int in_col = input_col_start + filter_col;

          const int input_offset =
              (input_offset_temp) + (in_row * in_width) + in_col;
          const int filter_offset =
              multiplier +
              depth_multiplier *
                  (in_channel + in_depth * (filter_col + filter_offset_temp));
          sum += static_cast<S>(__ldg(input + input_offset)) *
                 static_cast<S>(__ldg(filter + filter_offset));
        }
      }
    } else {
// Loop that needs to check for boundary conditions.
#pragma unroll
      for (int filter_row = 0; filter_row < filter_height; ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
#pragma unroll
        for (int filter_col = 0; filter_col < filter_width; ++filter_col) {
          const int in_col = input_col_start + filter_col;
          // TODO(vrv): the in_row check can be done outside of this loop;
          // benchmark both methods to determine the better decision.
          if (in_row >= 0 && in_row < in_height && in_col >= 0 &&
              in_col < in_width) {
            const int in_col = input_col_start + filter_col;

            // input_offset_temp indexes into the start of memory
            // where the spatial data starts.
            const int input_offset =
                (input_offset_temp) + (in_row * in_width) + in_col;

            const int filter_offset =
                multiplier +
                depth_multiplier *
                    (in_channel + in_depth * (filter_col + filter_offset_temp));
            sum += static_cast<S>(__ldg(input + input_offset)) *
                   static_cast<S>(__ldg(filter + filter_offset));
          }
        }
      }
    }

    output[thread_id] = static_cast<S>(sum);
  }
}
// Node name:	DepthwiseConv2dNative_834
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_832_0	type: float	shape: Shape{1, 64, 32, 32}
//	- name: Constant_346_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_834_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_834_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(128, 1, 1);
  const dim3 gridDim(128, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);

  typedef float S;
  float *input = input0;
  float *filter = input1;
  float *output = output0;

  const int in_height = 32;
  const int in_width = 32;
  const int in_depth = 64;
  const int filter_height = 5;
  const int filter_width = 5;
  const int depth_multiplier = 1;
  const int stride = 2;
  const int pad_height = 1;
  const int pad_width = 1;
  const int out_height = 16;
  const int out_width = 16;
  const int out_depth = 64;
  const int num_outputs = 16384;

  for (uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
       thread_id < num_outputs; thread_id += blockDim.x * gridDim.x) {
    // Compute the indexes of this thread in the output.
    //
    // We want coalesced reads so we make sure that each warp reads
    // a contiguous chunk of memory.
    //
    // THIS IS PROBABLY WRONG, we are not doing coalesced reads
    // into the input, because of the depth multiplier division...
    const int out_col = thread_id % out_width;
    const int out_row = (thread_id / out_width) % out_height;
    const int out_channel = (thread_id / out_width / out_height) % out_depth;
    const int batch = thread_id / out_width / out_height / out_depth;

    // Compute the input depth and the index of depth multiplier
    // based off the output depth index that this thread is
    // computing n.
    const int in_channel = out_channel / depth_multiplier;
    const int multiplier = out_channel % depth_multiplier;

    // Data is stored in the following format (let's assume we
    // flatten the height and width into one contiguous dimension
    // called "P".
    //
    // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
    //
    // Each row contains in_depth * in_height * in_width values
    // for each sample in the batch.
    //
    // We can further flatten it into:
    //
    // B1C1P1 B1C1P2 .....
    // B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 .....
    // B2C2P1 B2C2P2 ....
    //
    // where each row is a contiguous array of all of the spatial
    // pixels for a given batch and input depth.  The following
    // loop #pragma unrolls across the filter dimensions for a given thread,
    // indexing into the filter value and the corresponding input
    // patch.
    //
    // We can compute the index into the patch once right here.
    const int input_offset_temp =
        (batch * in_depth + in_channel) * (in_height * in_width);

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    //
    // We perform an additional optimization, where we can determine
    // whether the patch fits within the image indices statically, and
    // avoid boundary checking within the loop.
    const int input_row_start = out_row * stride - pad_height;
    const int input_col_start = out_col * stride - pad_width;
    const int input_row_end = input_row_start + filter_height;
    const int input_col_end = input_col_start + filter_width;

    S sum = static_cast<S>(0);
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < in_height && input_col_end < in_width) {
// Loop that doesn't need to check for boundary conditions.
#pragma unroll
      for (int filter_row = 0; filter_row < filter_height; ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
#pragma unroll
        for (int filter_col = 0; filter_col < filter_width; ++filter_col) {
          const int in_col = input_col_start + filter_col;

          const int input_offset =
              (input_offset_temp) + (in_row * in_width) + in_col;
          const int filter_offset =
              multiplier +
              depth_multiplier *
                  (in_channel + in_depth * (filter_col + filter_offset_temp));
          sum += static_cast<S>(__ldg(input + input_offset)) *
                 static_cast<S>(__ldg(filter + filter_offset));
        }
      }
    } else {
// Loop that needs to check for boundary conditions.
#pragma unroll
      for (int filter_row = 0; filter_row < filter_height; ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
#pragma unroll
        for (int filter_col = 0; filter_col < filter_width; ++filter_col) {
          const int in_col = input_col_start + filter_col;
          // TODO(vrv): the in_row check can be done outside of this loop;
          // benchmark both methods to determine the better decision.
          if (in_row >= 0 && in_row < in_height && in_col >= 0 &&
              in_col < in_width) {
            const int in_col = input_col_start + filter_col;

            // input_offset_temp indexes into the start of memory
            // where the spatial data starts.
            const int input_offset =
                (input_offset_temp) + (in_row * in_width) + in_col;

            const int filter_offset =
                multiplier +
                depth_multiplier *
                    (in_channel + in_depth * (filter_col + filter_offset_temp));
            sum += static_cast<S>(__ldg(input + input_offset)) *
                   static_cast<S>(__ldg(filter + filter_offset));
          }
        }
      }
    }

    output[thread_id] = static_cast<S>(sum);
  }
}
// Node name:	Add_2229
// Description:	Add
// Input:
//	- name: Convolution_829_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Constant_2740_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_831_0	type: float	shape: Shape{1,
// 32, 32, 32}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2229_block_kernel(float *input0, float *input1,
                                                 float *output0, int thread_id,
                                                 int block_id,
                                                 char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(64, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  output0[blockIdx.x * 512 + threadIdx.x] =
      add(input0[blockIdx.x * 512 + threadIdx.x],
          input1[blockIdx.x * 512 + threadIdx.x]);
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_46(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1, float *output2,
    float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Add_float_float_float_cuda_Add_2229_block_kernel(
        input4, input5, output3, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 191) {
    fused2_DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_836_block_kernel(
        input0, input1, input3, output0, output2, threadIdx.x, blockIdx.x - 64,
        NULL);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 319) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_834_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 192, NULL);
  }
  // else if((int)blockIdx.x >= 320 && (int)blockIdx.x <= 447)
  //{
  //    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_836_block_kernel(input0,
  //    input3, output2, threadIdx.x, blockIdx.x - 320, NULL);
  //}
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_46_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1, float *output2,
    float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_46<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1322_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2503_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_1323_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2506_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_1328_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Convolution_1330_0	type: float	shape: Shape{1, 64, 16,
// 16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1328<<<dim3(1, 16, 4), dim3(8,
// 1, 16), 0, 0>>>(DepthwiseConv2dNative_1322_0, Constant_2503_0,
// Convolution_1328_0);
// Convolution_float_float_float_cuda_Convolution_1330<<<dim3(1, 16, 4), dim3(8,
// 1, 16), 0, 0>>>(DepthwiseConv2dNative_1323_0, Constant_2506_0,
// Convolution_1330_0); Deduped function map: <src_function_name :
// deduped_function_name> Convolution_float_float_float_cuda_Convolution_1330 :
// Convolution_float_float_float_cuda_Convolution_1328

// Node name:	Convolution_1328
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1322_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2503_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_1328_0	type: float	shape: Shape{1, 64, 16,
// 16}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1328_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_117(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {

  __shared__ char shared_buffer[2048];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_1328_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_1328_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_117_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_117<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1287_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2485_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3110_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1285_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2479_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3112_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1286_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2482_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3114_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1282_0	type: float	shape: Shape{1,
// 64, 16, 16}
// Output:
//	- name: Relu_1308_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1306_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1307_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Slice_1284_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3109<<<dim3(1,
// 4, 8), dim3(4, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1287_0, Constant_2485_0,
// Constant_3110_0, Relu_1308_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3111<<<dim3(1,
// 4, 8), dim3(4, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1285_0, Constant_2479_0,
// Constant_3112_0, Relu_1306_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3113<<<dim3(1,
// 4, 8), dim3(4, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1286_0, Constant_2482_0,
// Constant_3114_0, Relu_1307_0); Slice_float_float_cuda_Slice_1284<<<dim3(256,
// 1, 1), dim3(64, 1, 1), 0, 0>>>(BatchNormInference_1282_0, Slice_1284_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3111 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3109
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3113 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3109

// Node name:	Matched_Pattern_3109
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1287_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2485_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3110_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1308_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3109_block_kernel(
    float *input0, float *input1, float *input2, float *output0, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(4, 2, 16);
  const dim3 gridDim(1, 4, 8);
  const dim3 threadIdx(thread_id % 4, thread_id / 4 % 2, thread_id / 8);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 4, block_id / 4);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 1024);
  {
    float *compute = output0;
    {
      float compute1[2];

      compute1[0] = 0.000000e+00f;
      compute1[1] = 0.000000e+00f;
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.y) * 8)) +
                  (((int)threadIdx.x) * 2))];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                   (((int)threadIdx.y) * 8)) +
                  (((int)threadIdx.x) * 2))];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      __syncthreads();
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                           (((int)threadIdx.x) * 2))] *
                          input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  1024)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  1025)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  16)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  17)];
      __syncthreads();
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                           (((int)threadIdx.x) * 2))] *
                          input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  2048)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  2049)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  32)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  33)];
      __syncthreads();
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                           (((int)threadIdx.x) * 2))] *
                          input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  3072)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  3073)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  48)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  49)];
      __syncthreads();
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                           (((int)threadIdx.x) * 2))] *
                          input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  4096)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  4097)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  64)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  65)];
      __syncthreads();
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                           (((int)threadIdx.x) * 2))] *
                          input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  5120)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  5121)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  80)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  81)];
      __syncthreads();
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                           (((int)threadIdx.x) * 2))] *
                          input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  6144)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  6145)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  96)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  97)];
      __syncthreads();
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                           (((int)threadIdx.x) * 2))] *
                          input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  7168)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  7169)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  112)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  113)];
      __syncthreads();
      compute1[0] =
          (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                           (((int)threadIdx.x) * 2))] *
                          input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                 (((int)blockIdx.y) * 16)) +
                (((int)threadIdx.y) * 8)) +
               (((int)threadIdx.x) * 2))] =
          max((compute1[0] +
               input2[((((int)blockIdx.z) * 16) + ((int)threadIdx.z))]),
              0.000000e+00f);
      compute[((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                  (((int)blockIdx.y) * 16)) +
                 (((int)threadIdx.y) * 8)) +
                (((int)threadIdx.x) * 2)) +
               1)] =
          max((compute1[1] +
               input2[((((int)blockIdx.z) * 16) + ((int)threadIdx.z))]),
              0.000000e+00f);
    }
  }
}
// Node name:	Slice_1284
// Description:	Slice
// Input:
//	- name: BatchNormInference_1282_0	type: float	shape: Shape{1,
// 64, 16, 16}
// Output:
//	- name: Slice_1284_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Slice_float_float_cuda_Slice_1284_block_kernel(float *input0, float *output0,
                                               int thread_id, int block_id,
                                               char *shared_buffer) {
  if (thread_id >= 64) {
    return;
  }
  const dim3 blockDim(64, 1, 1);
  const dim3 gridDim(256, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < 16384) {
    uint32_t input_strides[] = {16384, 256, 16, 1};
    uint32_t output_strides[] = {16384, 256, 16, 1};
    uint32_t lower_bounds[] = {0, 0, 0, 0};
    uint32_t slice_strides[] = {1, 1, 1, 1};
    uint32_t input_idx = 0;
    uint32_t output_idx = tid;
    input_idx += (((output_idx / output_strides[0]) * slice_strides[0]) +
                  lower_bounds[0]) *
                 input_strides[0];
    output_idx %= output_strides[0];
    input_idx += (((output_idx / output_strides[1]) * slice_strides[1]) +
                  lower_bounds[1]) *
                 input_strides[1];
    output_idx %= output_strides[1];
    input_idx += (((output_idx / output_strides[2]) * slice_strides[2]) +
                  lower_bounds[2]) *
                 input_strides[2];
    output_idx %= output_strides[2];
    input_idx += (((output_idx / output_strides[3]) * slice_strides[3]) +
                  lower_bounds[3]) *
                 input_strides[3];
    output0[tid] = input0[input_idx];
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Slice_112(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {

  __shared__ char shared_buffer[2048];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3109_block_kernel(
        input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3109_block_kernel(
        input3, input4, input5, output1, threadIdx.x, blockIdx.x - 32,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 95) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3109_block_kernel(
        input6, input7, input8, output2, threadIdx.x, blockIdx.x - 64,
        shared_buffer);
  } else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 351) {
    Slice_float_float_cuda_Slice_1284_block_kernel(
        input9, output3, threadIdx.x, blockIdx.x - 96, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Slice_112_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Slice_112<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, input8, input9,
                                    output0, output1, output2, output3);
}
