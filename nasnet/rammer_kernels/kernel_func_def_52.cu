// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	Constant_2107
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2107_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2107(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2107_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2107_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2839
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2839_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2839(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2839_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2839_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2533
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2533_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2533(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2533_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2533_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_72
// Description:	Constant
// Input:
// Output:
//	- name: Constant_72_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_72(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_72_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_72_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_258
// Description:	Constant
// Input:
// Output:
//	- name: Constant_258_0	type: float	shape: Shape{7, 7, 128, 1}
void Constant_float_cuda_Constant_258(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_258_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_258_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[25088];
  bin_file.read(tmp_mem, 25088);
  cudaMemcpyAsync(output0, tmp_mem, 25088, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_282
// Description:	Constant
// Input:
// Output:
//	- name: Constant_282_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_282(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_282_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_282_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2611
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2611_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2611(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2611_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2611_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3050
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3050_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3050(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3050_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3050_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_434
// Description:	Constant
// Input:
// Output:
//	- name: Constant_434_0	type: float	shape: Shape{5, 5, 96, 1}
void Constant_float_cuda_Constant_434(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_434_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_434_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[9600];
  bin_file.read(tmp_mem, 9600);
  cudaMemcpyAsync(output0, tmp_mem, 9600, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2814
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2814_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2814(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2814_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2814_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2164
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2164_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2164(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2164_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2164_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2170
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2170_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2170(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2170_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2170_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2239
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2239_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2239(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2239_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2239_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1150_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2401_0	type: float	shape: Shape{64, 384, 1, 1}
//	- name: Constant_2404_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1152_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Convolution_1154_0	type: float	shape: Shape{1, 64, 16,
//16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1152<<<dim3(1, 16, 4),
// dim3(16, 1, 16), 0, 0>>>(Relu_1150_0, Constant_2401_0, Convolution_1152_0);
// Convolution_float_float_float_cuda_Convolution_1154<<<dim3(1, 16, 4),
// dim3(16, 1, 16), 0, 0>>>(Relu_1150_0, Constant_2404_0, Convolution_1154_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1154 :
// Convolution_float_float_float_cuda_Convolution_1152

// Node name:	Convolution_1152
// Description:	Convolution
// Input:
//	- name: Relu_1150_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2401_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1152_0	type: float	shape: Shape{1, 64, 16,
//16}
__device__ __forceinline__ static void
Convolution_float_float_float_cuda_Convolution_1152_block_kernel(
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
          input0[((((((int)threadIdx.z) * 768) +
                    (((((int)threadIdx.x) * 3) / 16) * 256)) +
                   (((int)blockIdx.y) * 16)) +
                  ((((int)threadIdx.x) * 3) & 15))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          input0[((((((int)threadIdx.z) * 768) +
                    ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                   (((int)blockIdx.y) * 16)) +
                  (((((int)threadIdx.x) * 3) + 1) & 15))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          input0[((((((int)threadIdx.z) * 768) +
                    ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                   (((int)blockIdx.y) * 16)) +
                  (((((int)threadIdx.x) * 3) + 2) & 15))];
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
          input0[(((((((int)threadIdx.z) * 768) +
                     (((((int)threadIdx.x) * 3) / 16) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   ((((int)threadIdx.x) * 3) & 15)) +
                  12288)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 1) & 15)) +
                  12288)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 2) & 15)) +
                  12288)];
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
          input0[(((((((int)threadIdx.z) * 768) +
                     (((((int)threadIdx.x) * 3) / 16) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   ((((int)threadIdx.x) * 3) & 15)) +
                  24576)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 1) & 15)) +
                  24576)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 2) & 15)) +
                  24576)];
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
          input0[(((((((int)threadIdx.z) * 768) +
                     (((((int)threadIdx.x) * 3) / 16) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   ((((int)threadIdx.x) * 3) & 15)) +
                  36864)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 1) & 15)) +
                  36864)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 2) & 15)) +
                  36864)];
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
          input0[(((((((int)threadIdx.z) * 768) +
                     (((((int)threadIdx.x) * 3) / 16) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   ((((int)threadIdx.x) * 3) & 15)) +
                  49152)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 1) & 15)) +
                  49152)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 2) & 15)) +
                  49152)];
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
          input0[(((((((int)threadIdx.z) * 768) +
                     (((((int)threadIdx.x) * 3) / 16) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   ((((int)threadIdx.x) * 3) & 15)) +
                  61440)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 1) & 15)) +
                  61440)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 2) & 15)) +
                  61440)];
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
          input0[(((((((int)threadIdx.z) * 768) +
                     (((((int)threadIdx.x) * 3) / 16) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   ((((int)threadIdx.x) * 3) & 15)) +
                  73728)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 1) & 15)) +
                  73728)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 2) & 15)) +
                  73728)];
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
          input0[(((((((int)threadIdx.z) * 768) +
                     (((((int)threadIdx.x) * 3) / 16) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   ((((int)threadIdx.x) * 3) & 15)) +
                  86016)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 1) & 15)) +
                  86016)];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          input0[(((((((int)threadIdx.z) * 768) +
                     ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                    (((int)blockIdx.y) * 16)) +
                   (((((int)threadIdx.x) * 3) + 2) & 15)) +
                  86016)];
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
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_92(
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  __shared__ char shared_buffer[6144];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_1152_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_1152_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 64 + 0,
        shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_92_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_92<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_638_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2824_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2822_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_640_0	type: float	shape: Shape{1, 32, 32,
//32}
// Output:
//	- name: BatchNormInference_641_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: BatchNormInference_642_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Relu_644_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2118<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_638_0, Constant_2824_0, BatchNormInference_641_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_11<<<dim3(64, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_640_0, Constant_2822_0, Relu_644_0,
// BatchNormInference_642_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	Add_2118
// Description:	Add
// Input:
//	- name: Convolution_638_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2824_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_641_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ static void
Add_float_float_float_cuda_Add_2118_block_kernel(float *input0, float *input1,
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
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_640_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2822_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_644_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_642_0	type: float	shape: Shape{1,
//32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2121<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_640_0, Constant_2822_0, BatchNormInference_642_0);
// Relu_float_float_cuda_Relu_644<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_642_0, Relu_644_0);
__device__ __forceinline__ static void
FusedKernel_float_float_float_float_cuda_Add_Relu_11_block_kernel(
    float *input0, float *input1, float *output0, float *output1, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(64, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = relu(temp0);
  output1[tid] = temp0;
  output0[tid] = temp1;
}
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_19(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Add_float_float_float_cuda_Add_2118_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    FusedKernel_float_float_float_float_cuda_Add_Relu_11_block_kernel(
        input3, input2, output2, output1, threadIdx.x, blockIdx.x - 64 + 0,
        NULL);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_19_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {
  BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_19<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1, output2);
}
