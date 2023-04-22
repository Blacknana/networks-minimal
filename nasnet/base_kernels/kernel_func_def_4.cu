// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	Constant_2864
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2864_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2864(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2864_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2864_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2221
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2221_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2221(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2221_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2221_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2386
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2386_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2386(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2386_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2386_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2476
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2476_0	type: float	shape: Shape{64, 384, 1, 1}
void Constant_float_cuda_Constant_2476(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2476_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2476_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[98304];
  bin_file.read(tmp_mem, 98304);
  cudaMemcpyAsync(output0, tmp_mem, 98304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3116
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3116_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3116(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3116_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3116_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_117
// Description:	Constant
// Input:
// Output:
//	- name: Constant_117_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_117(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_117_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_117_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3126
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3126_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3126(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3126_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3126_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2218
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2218_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2218(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2218_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2218_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2236
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2236_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2236(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2236_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2236_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3120
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3120_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3120(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3120_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3120_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_167
// Description:	Constant
// Input:
// Output:
//	- name: Constant_167_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_167(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_167_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_167_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2350
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2350_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2350(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2350_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2350_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_961_0	type: float	shape: Shape{1, 256, 16, 16}
//	- name: Constant_2293_0	type: float	shape: Shape{64, 256, 1, 1}
//	- name: Constant_2296_0	type: float	shape: Shape{64, 256, 1, 1}
// Output:
//	- name: Convolution_963_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Convolution_965_0	type: float	shape: Shape{1, 64, 16,
// 16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_963<<<dim3(1, 16, 4), dim3(16,
// 1, 16), 0, 0>>>(Relu_961_0, Constant_2293_0, Convolution_963_0);
// Convolution_float_float_float_cuda_Convolution_965<<<dim3(1, 16, 4), dim3(16,
// 1, 16), 0, 0>>>(Relu_961_0, Constant_2296_0, Convolution_965_0); Deduped
// function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_965 :
// Convolution_float_float_float_cuda_Convolution_963

// Node name:	Convolution_963
// Description:	Convolution
// Input:
//	- name: Relu_961_0	type: float	shape: Shape{1, 256, 16, 16}
//	- name: Constant_2293_0	type: float	shape: Shape{64, 256, 1, 1}
// Output:
//	- name: Convolution_963_0	type: float	shape: Shape{1, 64, 16,
// 16}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_963_block_kernel(
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
  float *input1_shared = (float *)(shared_buffer + 1024);
  {
    float *compute = output0;
    {
      float compute_local[1];

      compute_local[0] = 0.000000e+00f;
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          relu(input0[(((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                       ((int)threadIdx.x))]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[(((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                  ((int)threadIdx.x))];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  4096)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  16)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  8192)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  32)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  12288)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  48)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  16384)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  64)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  20480)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  80)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  24576)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  96)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  28672)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  112)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  32768)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  128)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  36864)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  144)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  40960)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  160)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  45056)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  176)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  49152)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  192)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  53248)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  208)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  57344)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  224)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  61440)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  240)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                (((int)blockIdx.y) * 16)) +
               ((int)threadIdx.x))] = compute_local[0];
    }
  }
}
__device__ __forceinline__ void
fuse2_Convolution_float_float_float_cuda_Convolution_963_block_kernel(
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
  float *input1_shared = (float *)(shared_buffer + 1024);
  float *input1_shared1 = (float *)(shared_buffer + 2048);
  {
    float *compute = output0;
    {
      float compute_local[1];
      float compute_local1[1];

      compute_local[0] = 0.000000e+00f;
      compute_local1[0] = 0.000000e+00f;
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          relu(input0[(((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                       ((int)threadIdx.x))]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[(((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                  ((int)threadIdx.x))];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[(((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                  ((int)threadIdx.x))];

      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));

      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  4096)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  16)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  16)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  8192)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  32)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  32)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));

      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  12288)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  48)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  48)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));

      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  16384)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  64)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  64)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  20480)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  80)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  80)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  24576)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  96)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  96)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  28672)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  112)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  112)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  32768)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  128)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  128)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  36864)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  144)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  144)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  40960)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  160)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  160)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  45056)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  176)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  176)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  49152)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  192)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  192)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  53248)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  208)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  208)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  57344)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  224)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  224)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = relu(
          input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) +
                   ((int)threadIdx.x)) +
                  61440)]);
      input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  240)];
      input1_shared1[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] =
          input2[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   ((int)threadIdx.x)) +
                  240)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                               input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 16)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 16)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 1)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 2)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 48)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 3)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 4)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 80)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 5)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 6)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 112)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 7)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 8)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 144)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 9)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 10)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 176)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 11)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 12)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 208)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 13)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 14)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 240)] *
                            input1_shared1[((((int)threadIdx.z) * 16) + 15)]));
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
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_65(
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {

  // __shared__ char shared_buffer[2048];
  __shared__ char shared_buffer[3072];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    fuse2_Convolution_float_float_float_cuda_Convolution_963_block_kernel(
        input0, input1, input2, output0, output1, threadIdx.x, blockIdx.x - 0,
        shared_buffer);
    // Convolution_float_float_float_cuda_Convolution_963_block_kernel(input0,
    // input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  }
  // else if((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
  // {
  //     Convolution_float_float_float_cuda_Convolution_963_block_kernel(input0,
  //     input2, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  // }
}

extern void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_65_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_65<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	AvgPool_1361
// Description:	AvgPool
// Input:
//	- name: BatchNormInference_1357_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Output:
//	- name: AvgPool_1361_0	type: float	shape: Shape{1, 128, 8, 8}
void AvgPool_float_float_cuda_lib_AvgPool_1361(cudnnHandle_t cudnn_handle,
                                               float *input0, float *output0) {
  cudnnTensorDescriptor_t input_desc;
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, 1, 128, 8, 8));
  cudnnTensorDescriptor_t output_desc;
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
  CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, 1, 128, 8, 8));
  cudnnPoolingDescriptor_t desc;
  cudnnCreatePoolingDescriptor(&desc);
  CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(
      desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
      CUDNN_NOT_PROPAGATE_NAN, 3, 3, 1, 1, 1, 1));
  const float alpha = 1.0;
  const float beta = 0.0;
  CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc,
                                      input0, &beta, output_desc, output0));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
  CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));
}
// Node name:	AvgPool_1361
// Description:	AvgPool
// Input:
//	- name: BatchNormInference_1357_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Output:
//	- name: AvgPool_1361_0	type: float	shape: Shape{1, 128, 8, 8}
// 3, 1, 1(kernelH, pad, stride)
// grid(32,1,1) block(256,1,1)
extern "C" __global__ void operator_avg_pool_h_128_8_8_3x3_1(const float *input,
                                                             float *output) {

  const int pooled_height = 8;
  const int pooled_width = 8;
  const int nthreads = 8192;
  int index = blockIdx.x * 256 + threadIdx.x;

  if (index < nthreads) {
    const int kChannels = 128;
    const int kHeight = 8;
    const int kWidth = 8;
    const int kKernelH = 3;
    const int kKernelW = 3;
    const int kStrideH = 1;
    const int kStrideW = 1;
    const int kPadH = 1;
    const int kPadW = 1;

    // output location
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % kChannels;
    const int n = index / pooled_width / pooled_height / kChannels;

    // pooled range
    int hstart = ph * kStrideH - kPadH;
    int wstart = pw * kStrideW - kPadW;
    const int hend = fminf(hstart + kKernelH, kHeight);
    const int wend = fminf(wstart + kKernelW, kWidth);
    hstart = fmaxf(hstart, 0);
    wstart = fmaxf(wstart, 0);

    float avgval = 0.0f;
    int slice_offset = (n * kChannels + c) * kHeight * kWidth;
#pragma unroll 4
    for (int h = hstart; h < hend; ++h) {
#pragma unroll 4
      for (int w = wstart; w < wend; ++w) {
        avgval = (input[slice_offset + h * kWidth + w]) /
                     ((hend - hstart) * (wend - wstart)) +
                 avgval;
      }
    }

    // output
    output[index] = avgval;
  }
}

extern void BlockFusionKernel_2_AvgPool_1361_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *output0, float *output1) {
  operator_avg_pool_h_128_8_8_3x3_1<<<32, blocks, mem, stream>>>(input0,
                                                                 output0);
  operator_avg_pool_h_128_8_8_3x3_1<<<32, blocks, mem, stream>>>(input1,
                                                                 output1);
}

// Node name:	Constant_3180
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3180_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3180(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3180_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3180_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
