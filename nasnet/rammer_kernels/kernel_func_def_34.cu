// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_27
// Description:	Constant
// Input:
// Output:
//	- name: Constant_27_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_27(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_27_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_27_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2551
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2551_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2551(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2551_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2551_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3092
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3092_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3092(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3092_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3092_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2758
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2758_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2758(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2758_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2758_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2065
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2065_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2065(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2065_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2065_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_1852
// Description:	Constant
// Input:
// Output:
//	- name: Constant_1852_0	type: float	shape: Shape{32, 192, 1, 1}
void Constant_float_cuda_Constant_1852(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_1852_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_1852_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[24576];
  bin_file.read(tmp_mem, 24576);
  cudaMemcpyAsync(output0, tmp_mem, 24576, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3010
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3010_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_3010(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3010_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3010_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2914
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2914_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2914(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2914_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2914_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2242
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2242_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2242(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2242_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2242_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2338
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2338_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2338(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2338_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2338_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_303
// Description:	Constant
// Input:
// Output:
//	- name: Constant_303_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_303(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_303_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_303_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_573_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2080_0	type: float	shape: Shape{32, 192, 1, 1}
//	- name: Constant_2083_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_577_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Convolution_575_0	type: float	shape: Shape{1, 32, 32,
//32}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_577<<<dim3(1, 32, 2), dim3(32,
// 1, 8), 0, 0>>>(Relu_573_0, Constant_2080_0, Convolution_577_0);
// Convolution_float_float_float_cuda_Convolution_575<<<dim3(1, 32, 2), dim3(32,
// 1, 8), 0, 0>>>(Relu_573_0, Constant_2083_0, Convolution_575_0); Deduped
// function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_575 :
// Convolution_float_float_float_cuda_Convolution_577

// Node name:	Convolution_577
// Description:	Convolution
// Input:
//	- name: Relu_573_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2080_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_577_0	type: float	shape: Shape{1, 32, 32,
//32}
__device__ __forceinline__ static void
Convolution_float_float_float_cuda_Convolution_577_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(32, 1, 8);
  const dim3 gridDim(1, 32, 2);
  const dim3 threadIdx(thread_id % 32, 0, thread_id / 32);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 32, block_id / 32);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 6144);
  {
    float *compute = output0;
    {
      float compute_local[2];

      compute_local[0] = 0.000000e+00f;
      compute_local[1] = 0.000000e+00f;
      pad_temp_shared[((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6))] =
          input0[((((((int)threadIdx.z) * 6144) +
                    (((((int)threadIdx.x) * 6) / 32) * 1024)) +
                   (((int)blockIdx.y) * 32)) +
                  ((((int)threadIdx.x) * 6) & 31))];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 6144) +
                    ((((((int)threadIdx.x) * 6) + 1) >> 5) * 1024)) +
                   (((int)blockIdx.y) * 32)) +
                  (((((int)threadIdx.x) * 6) + 1) & 31))];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       2)] =
          input0[((((((int)threadIdx.z) * 6144) +
                    ((((((int)threadIdx.x) * 6) + 2) >> 5) * 1024)) +
                   (((int)blockIdx.y) * 32)) +
                  (((((int)threadIdx.x) * 6) + 2) & 31))];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       3)] =
          input0[((((((int)threadIdx.z) * 6144) +
                    ((((((int)threadIdx.x) * 6) + 3) >> 5) * 1024)) +
                   (((int)blockIdx.y) * 32)) +
                  (((((int)threadIdx.x) * 6) + 3) & 31))];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       4)] =
          input0[((((((int)threadIdx.z) * 6144) +
                    ((((((int)threadIdx.x) * 6) + 4) >> 5) * 1024)) +
                   (((int)blockIdx.y) * 32)) +
                  (((((int)threadIdx.x) * 6) + 4) & 31))];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       5)] =
          input0[((((((int)threadIdx.z) * 6144) +
                    ((((((int)threadIdx.x) * 6) + 5) >> 5) * 1024)) +
                   (((int)blockIdx.y) * 32)) +
                  (((((int)threadIdx.x) * 6) + 5) & 31))];
      input1_shared[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                   ((((int)threadIdx.x) >> 4) * 192)) +
                  ((((int)threadIdx.x) & 15) * 3))];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                   ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) +
                  (((((int)threadIdx.x) * 3) + 1) % 48))];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                   ((((((int)threadIdx.x) * 3) + 2) / 48) * 192)) +
                  (((((int)threadIdx.x) * 3) + 2) % 48))];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 96)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((int)threadIdx.x)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 49)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 50)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 51)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 52)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 53)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 54)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 55)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 56)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 57)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 58)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 59)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 60)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 61)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 62)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 63)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 16)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 64)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 17)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 65)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 18)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 66)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 19)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 67)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 20)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 68)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 21)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 69)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 22)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 70)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 23)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 71)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 24)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 72)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 25)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 73)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 26)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 74)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 27)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 75)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 28)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 76)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 29)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 77)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 30)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 78)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 31)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 79)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 32)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 80)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 33)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 81)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 34)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 82)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 35)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 83)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 36)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 84)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 37)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 85)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 38)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 86)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 39)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 87)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 40)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 88)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 41)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 89)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 42)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 90)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 43)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 91)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 44)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 92)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 45)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 93)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 46)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 94)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 47)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 95)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6))] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     (((((int)threadIdx.x) * 6) / 32) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   ((((int)threadIdx.x) * 6) & 31)) +
                  49152)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 1) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 1) & 31)) +
                  49152)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       2)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 2) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 2) & 31)) +
                  49152)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       3)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 3) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 3) & 31)) +
                  49152)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       4)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 4) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 4) & 31)) +
                  49152)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       5)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 5) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 5) & 31)) +
                  49152)];
      input1_shared[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((int)threadIdx.x) >> 4) * 192)) +
                   ((((int)threadIdx.x) & 15) * 3)) +
                  48)];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 1) % 48)) +
                  48)];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 2) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 2) % 48)) +
                  48)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 96)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((int)threadIdx.x)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 49)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 50)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 51)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 52)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 53)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 54)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 55)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 56)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 57)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 58)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 59)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 60)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 61)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 62)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 63)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 16)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 64)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 17)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 65)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 18)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 66)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 19)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 67)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 20)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 68)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 21)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 69)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 22)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 70)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 23)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 71)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 24)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 72)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 25)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 73)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 26)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 74)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 27)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 75)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 28)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 76)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 29)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 77)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 30)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 78)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 31)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 79)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 32)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 80)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 33)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 81)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 34)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 82)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 35)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 83)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 36)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 84)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 37)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 85)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 38)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 86)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 39)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 87)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 40)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 88)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 41)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 89)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 42)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 90)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 43)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 91)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 44)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 92)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 45)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 93)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 46)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 94)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 47)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 95)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6))] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     (((((int)threadIdx.x) * 6) / 32) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   ((((int)threadIdx.x) * 6) & 31)) +
                  98304)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 1) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 1) & 31)) +
                  98304)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       2)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 2) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 2) & 31)) +
                  98304)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       3)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 3) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 3) & 31)) +
                  98304)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       4)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 4) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 4) & 31)) +
                  98304)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       5)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 5) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 5) & 31)) +
                  98304)];
      input1_shared[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((int)threadIdx.x) >> 4) * 192)) +
                   ((((int)threadIdx.x) & 15) * 3)) +
                  96)];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 1) % 48)) +
                  96)];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 2) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 2) % 48)) +
                  96)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 96)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((int)threadIdx.x)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 49)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 50)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 51)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 52)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 53)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 54)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 55)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 56)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 57)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 58)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 59)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 60)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 61)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 62)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 63)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 16)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 64)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 17)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 65)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 18)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 66)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 19)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 67)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 20)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 68)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 21)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 69)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 22)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 70)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 23)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 71)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 24)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 72)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 25)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 73)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 26)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 74)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 27)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 75)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 28)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 76)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 29)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 77)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 30)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 78)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 31)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 79)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 32)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 80)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 33)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 81)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 34)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 82)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 35)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 83)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 36)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 84)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 37)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 85)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 38)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 86)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 39)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 87)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 40)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 88)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 41)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 89)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 42)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 90)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 43)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 91)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 44)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 92)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 45)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 93)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 46)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 94)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 47)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 95)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6))] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     (((((int)threadIdx.x) * 6) / 32) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   ((((int)threadIdx.x) * 6) & 31)) +
                  147456)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 1) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 1) & 31)) +
                  147456)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       2)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 2) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 2) & 31)) +
                  147456)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       3)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 3) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 3) & 31)) +
                  147456)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       4)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 4) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 4) & 31)) +
                  147456)];
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       5)] =
          input0[(((((((int)threadIdx.z) * 6144) +
                     ((((((int)threadIdx.x) * 6) + 5) >> 5) * 1024)) +
                    (((int)blockIdx.y) * 32)) +
                   (((((int)threadIdx.x) * 6) + 5) & 31)) +
                  147456)];
      input1_shared[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((int)threadIdx.x) >> 4) * 192)) +
                   ((((int)threadIdx.x) & 15) * 3)) +
                  144)];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 1) % 48)) +
                  144)];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 2) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 2) % 48)) +
                  144)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 96)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((int)threadIdx.x)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 49)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 50)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 51)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 52)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 53)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 54)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 55)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 56)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 57)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 58)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 59)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 60)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 61)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 62)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 63)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 16)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 64)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 17)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 65)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 18)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 66)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 19)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 67)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 20)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 68)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 21)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 69)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 22)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 70)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 23)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 71)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 24)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 72)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 25)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 73)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 26)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 74)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 27)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 75)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 28)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 76)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 29)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 77)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 30)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 78)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 31)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 79)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 32)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 80)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 33)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 81)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 34)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 82)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 35)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 83)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 36)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 84)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 37)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 85)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 38)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 86)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 39)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 87)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 40)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 88)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 41)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 89)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 42)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 90)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 43)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 91)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 44)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 92)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 45)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 93)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 46)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 94)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 47)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 95)]));
      compute[((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) +
                (((int)blockIdx.y) * 32)) +
               ((int)threadIdx.x))] = compute_local[0];
      compute[(((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) +
                 (((int)blockIdx.y) * 32)) +
                ((int)threadIdx.x)) +
               1024)] = compute_local[1];
    }
  }
}
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_9(
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  __shared__ char shared_buffer[9216];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_577_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_577_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 64 + 0,
        shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_9_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_9<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_653_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2131_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3012_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_654_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2134_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3014_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_677_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2143_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_675_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2137_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_676_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2140_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Relu_673_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_674_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_685_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Convolution_681_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Convolution_683_0	type: float	shape: Shape{1, 32, 32,
//32}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3011<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_653_0, Constant_2131_0,
// Constant_3012_0, Relu_673_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3013<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_654_0, Constant_2134_0,
// Constant_3014_0, Relu_674_0);
// Convolution_float_float_float_cuda_Convolution_685<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_677_0, Constant_2143_0,
// Convolution_685_0);
// Convolution_float_float_float_cuda_Convolution_681<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_675_0, Constant_2137_0,
// Convolution_681_0);
// Convolution_float_float_float_cuda_Convolution_683<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_676_0, Constant_2140_0,
// Convolution_683_0); Deduped function map: <src_function_name :
// deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3013 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3011
// Convolution_float_float_float_cuda_Convolution_681 :
// Convolution_float_float_float_cuda_Convolution_685
// Convolution_float_float_float_cuda_Convolution_683 :
// Convolution_float_float_float_cuda_Convolution_685

// Node name:	Matched_Pattern_3011
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_653_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2131_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3012_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_673_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ static void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3011_block_kernel(
    float *input0, float *input1, float *input2, float *output0, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(16, 2, 8);
  const dim3 gridDim(2, 16, 2);
  const dim3 threadIdx(thread_id % 16, thread_id / 16 % 2, thread_id / 32);
  const dim3 blockIdx(block_id % 2, block_id / 2 % 16, block_id / 32);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 2048);
  {
    float *compute = output0;
    {
      float compute1[2];

      for (int ff_init = 0; ff_init < 2; ++ff_init) {
        compute1[ff_init] = 0.000000e+00f;
      }
      for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
        __syncthreads();
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
          pad_temp_shared[(
              (((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
               (((int)threadIdx.x) * 2)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] =
              input0[(
                  ((((((rc_outer * 16384) + (((int)threadIdx.z) * 2048)) +
                      (((int)threadIdx.y) * 1024)) +
                     (((int)blockIdx.y) * 64)) +
                    ((((((int)threadIdx.x) * 2) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >>
                      4) *
                     32)) +
                   (((int)blockIdx.x) * 16)) +
                  (((((int)threadIdx.x) * 2) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) &
                   15))];
        }
        input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       ((int)threadIdx.x))] =
            input1[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 64)) +
                      (((int)threadIdx.y) * 32)) +
                     (rc_outer * 16)) +
                    ((int)threadIdx.x))];
        __syncthreads();
        for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
          for (int ff = 0; ff < 2; ++ff) {
            compute1[ff] =
                (compute1[ff] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.y) * 16)) +
                      ((int)threadIdx.x))] *
                  input1_shared[(((((int)threadIdx.z) * 32) + (ff * 16)) +
                                 rc_inner)]));
          }
        }
      }
      for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 2;
           ++i1_inner_inner_inner) {
        compute[(
            ((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) +
                (i1_inner_inner_inner * 1024)) +
               (((int)blockIdx.y) * 64)) +
              (((int)threadIdx.y) * 32)) +
             (((int)blockIdx.x) * 16)) +
            ((int)threadIdx.x))] =
            max((compute1[i1_inner_inner_inner] +
                 input2[(((((int)blockIdx.z) * 16) + (((int)threadIdx.z) * 2)) +
                         i1_inner_inner_inner)]),
                0.000000e+00f);
      }
    }
  }
}
// Node name:	Convolution_685
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_677_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2143_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_685_0	type: float	shape: Shape{1, 32, 32,
//32}
__device__ __forceinline__ static void
Convolution_float_float_float_cuda_Convolution_685_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(16, 2, 8);
  const dim3 gridDim(2, 16, 2);
  const dim3 threadIdx(thread_id % 16, thread_id / 16 % 2, thread_id / 32);
  const dim3 blockIdx(block_id % 2, block_id / 2 % 16, block_id / 32);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 2048);
  {
    float *compute = output0;
    {
      float compute_local[2];

      for (int ff_c_init = 0; ff_c_init < 2; ++ff_c_init) {
        compute_local[ff_c_init] = 0.000000e+00f;
      }
      for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
        __syncthreads();
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
          pad_temp_shared[(
              (((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
               (((int)threadIdx.x) * 2)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] =
              input0[(
                  ((((((rc_outer * 16384) + (((int)threadIdx.z) * 2048)) +
                      (((int)threadIdx.y) * 1024)) +
                     (((int)blockIdx.y) * 64)) +
                    ((((((int)threadIdx.x) * 2) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >>
                      4) *
                     32)) +
                   (((int)blockIdx.x) * 16)) +
                  (((((int)threadIdx.x) * 2) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) &
                   15))];
        }
        input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       ((int)threadIdx.x))] =
            input1[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 64)) +
                      (((int)threadIdx.y) * 32)) +
                     (rc_outer * 16)) +
                    ((int)threadIdx.x))];
        __syncthreads();
        for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
          for (int ff_c = 0; ff_c < 2; ++ff_c) {
            compute_local[ff_c] =
                (compute_local[ff_c] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.y) * 16)) +
                      ((int)threadIdx.x))] *
                  input1_shared[(((((int)threadIdx.z) * 32) + (ff_c * 16)) +
                                 rc_inner)]));
          }
        }
      }
      for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 2;
           ++ff_inner_inner_inner) {
        compute[(
            ((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) +
                (ff_inner_inner_inner * 1024)) +
               (((int)blockIdx.y) * 64)) +
              (((int)threadIdx.y) * 32)) +
             (((int)blockIdx.x) * 16)) +
            ((int)threadIdx.x))] = compute_local[ff_inner_inner_inner];
      }
    }
  }
}
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_23(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {
  __shared__ char shared_buffer[3072];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3011_block_kernel(
        input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3011_block_kernel(
        input3, input4, input5, output1, threadIdx.x, blockIdx.x - 64 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    Convolution_float_float_float_cuda_Convolution_685_block_kernel(
        input6, input7, output2, threadIdx.x, blockIdx.x - 128 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    Convolution_float_float_float_cuda_Convolution_685_block_kernel(
        input8, input9, output3, threadIdx.x, blockIdx.x - 192 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319) {
    Convolution_float_float_float_cuda_Convolution_685_block_kernel(
        input10, input11, output4, threadIdx.x, blockIdx.x - 256 + 0,
        shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_23_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_23<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, input11, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2796_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1477_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Convolution_1479_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2755_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1480_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Relu_1483_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1481_0	type: float	shape: Shape{1,
//128, 8, 8}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Relu_57<<<dim3(16, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1477_0, Constant_2796_0, Relu_1483_0,
// BatchNormInference_1480_0); Add_float_float_float_cuda_Add_2583<<<dim3(16, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1479_0, Constant_2755_0,
// BatchNormInference_1481_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1477_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2796_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1483_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1480_0	type: float	shape: Shape{1,
//128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2580<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1477_0, Constant_2796_0, BatchNormInference_1480_0);
// Relu_float_float_cuda_Relu_1483<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1480_0, Relu_1483_0);
__device__ __forceinline__ static void
FusedKernel_float_float_float_float_cuda_Add_Relu_57_block_kernel(
    float *input0, float *input1, float *output0, float *output1, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(16, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = relu(temp0);
  output1[tid] = temp0;
  output0[tid] = temp1;
}
// Node name:	Add_2583
// Description:	Add
// Input:
//	- name: Convolution_1479_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2755_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1481_0	type: float	shape: Shape{1,
//128, 8, 8}
__device__ __forceinline__ static void
Add_float_float_float_cuda_Add_2583_block_kernel(float *input0, float *input1,
                                                 float *output0, int thread_id,
                                                 int block_id,
                                                 char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(16, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  output0[blockIdx.x * 512 + threadIdx.x] =
      add(input0[blockIdx.x * 512 + threadIdx.x],
          input1[blockIdx.x * 512 + threadIdx.x]);
}
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_140(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    FusedKernel_float_float_float_float_cuda_Add_Relu_57_block_kernel(
        input1, input0, output1, output0, threadIdx.x, blockIdx.x - 0 + 0,
        NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_2583_block_kernel(
        input2, input3, output2, threadIdx.x, blockIdx.x - 16 + 0, NULL);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_140_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {
  BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_140<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1, output2);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_912_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_156_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: BatchNormInference_908_0	type: float	shape: Shape{1,
//64, 16, 16}
// Output:
//	- name: DepthwiseConv2dNative_917_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Relu_910_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_917<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_912_0, Constant_156_0,
// DepthwiseConv2dNative_917_0); Relu_float_float_cuda_Relu_910<<<dim3(32, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_908_0, Relu_910_0); Deduped
// function map: <src_function_name : deduped_function_name>

// Node name:	DepthwiseConv2dNative_917
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_912_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_156_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_917_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ static void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_917_block_kernel(
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

  const int in_height = 16;
  const int in_width = 16;
  const int in_depth = 64;
  const int filter_height = 5;
  const int filter_width = 5;
  const int depth_multiplier = 1;
  const int stride = 1;
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
// Node name:	Relu_910
// Description:	Relu
// Input:
//	- name: BatchNormInference_908_0	type: float	shape: Shape{1,
//64, 16, 16}
// Output:
//	- name: Relu_910_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ static void
Relu_float_float_cuda_Relu_910_block_kernel(float *input0, float *output0,
                                            int thread_id, int block_id,
                                            char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(32, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  output0[blockIdx.x * 512 + threadIdx.x] =
      relu(input0[blockIdx.x * 512 + threadIdx.x]);
}
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_cuda_DepthwiseConv2dNative_Relu_59(
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_917_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 159) {
    Relu_float_float_cuda_Relu_910_block_kernel(input2, output1, threadIdx.x,
                                                blockIdx.x - 128 + 0, NULL);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_cuda_DepthwiseConv2dNative_Relu_59_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_cuda_DepthwiseConv2dNative_Relu_59<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
