// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_192
// Description:	Constant
// Input:
// Output:
//	- name: Constant_192_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_192(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_192_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_192_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2524
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2524_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2524(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2524_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2524_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3174
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3174_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3174(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3174_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3174_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_349
// Description:	Constant
// Input:
// Output:
//	- name: Constant_349_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_349(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_349_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_349_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_462
// Description:	Constant
// Input:
// Output:
//	- name: Constant_462_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_462(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_462_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_462_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_324
// Description:	Constant
// Input:
// Output:
//	- name: Constant_324_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_324(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_324_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_324_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3090
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3090_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3090(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3090_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3090_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3066
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3066_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3066(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3066_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3066_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2775
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2775_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2775(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2775_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2775_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_355
// Description:	Constant
// Input:
// Output:
//	- name: Constant_355_0	type: float	shape: Shape{3, 3, 96, 1}
void Constant_float_cuda_Constant_355(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_355_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_355_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[3456];
  bin_file.read(tmp_mem, 3456);
  cudaMemcpyAsync(output0, tmp_mem, 3456, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_112
// Description:	Constant
// Input:
// Output:
//	- name: Constant_112_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_112(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_112_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_112_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Convolution_957
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_955_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2290_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_957_0	type: float	shape: Shape{1, 64, 16,
//16}
extern "C" __global__ void
Convolution_float_float_float_cuda_Convolution_957(float *input0, float *input1,
                                                   float *output0) {
  __shared__ float pad_temp_shared[256];
  __shared__ float input1_shared[256];
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
extern void Convolution_float_float_float_cuda_Convolution_957_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *output0) {
  Convolution_float_float_float_cuda_Convolution_957<<<grids, blocks, mem,
                                                       stream>>>(input0, input1,
                                                                 output0);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2847_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_877_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: BatchNormInference_817_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2860_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_879_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Slice_833_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_882_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_883_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_23<<<dim3(64, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_877_0, Constant_2847_0,
// BatchNormInference_817_0, Add_882_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_24<<<dim3(64, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_879_0, Constant_2860_0, Slice_833_0,
// Add_883_0); Deduped function map: <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_24 :
// FusedKernel_float_float_float_float_cuda_Add_Add_23

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_877_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2847_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_817_0	type: float	shape: Shape{1,
//32, 32, 32}
// Output:
//	- name: Add_882_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2256<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_877_0, Constant_2847_0, BatchNormInference_880_0);
// Add_float_float_float_cuda_Add_882<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_880_0, BatchNormInference_817_0, Add_882_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Add_23_block_kernel(
    float *input0, float *input1, float *input2, float *output0, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(64, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = add(temp0, input2[tid]);
  output0[tid] = temp1;
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_53(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    FusedKernel_float_float_float_float_cuda_Add_Add_23_block_kernel(
        input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    FusedKernel_float_float_float_float_cuda_Add_Add_23_block_kernel(
        input4, input3, input5, output1, threadIdx.x, blockIdx.x - 64, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_53_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_53<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: AvgPool_1337_0	type: float	shape: Shape{1, 384, 8, 8}
//	- name: Constant_1935_0	type: float	shape: Shape{64, 384, 1, 1}
//	- name: Pad_1339_0	type: float	shape: Shape{1, 384, 17, 17}
//	- name: Convolution_1341_0	type: float	shape: Shape{1, 128, 16,
//16}
//	- name: Constant_2928_0	type: float	shape: Shape{1, 128, 16, 16}
// Output:
//	- name: Convolution_1343_0	type: float	shape: Shape{1, 64, 8,
//8}
//	- name: Slice_1344_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: BatchNormInference_1345_0	type: float	shape: Shape{1,
//128, 16, 16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1343<<<dim3(1, 8, 8), dim3(8,
// 1, 8), 0, 0>>>(AvgPool_1337_0, Constant_1935_0, Convolution_1343_0);
// Slice_float_float_cuda_Slice_1344<<<dim3(1536, 1, 1), dim3(64, 1, 1), 0,
// 0>>>(Pad_1339_0, Slice_1344_0);
// Add_float_float_float_cuda_Add_2511<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1341_0, Constant_2928_0, BatchNormInference_1345_0); Deduped
// function map: <src_function_name : deduped_function_name>

// Node name:	Convolution_1343
// Description:	Convolution
// Input:
//	- name: AvgPool_1337_0	type: float	shape: Shape{1, 384, 8, 8}
//	- name: Constant_1935_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1343_0	type: float	shape: Shape{1, 64, 8,
//8}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1343_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 64) {
    return;
  }
  const dim3 blockDim(8, 1, 8);
  const dim3 gridDim(1, 8, 8);
  const dim3 threadIdx(thread_id % 8, 0, thread_id / 8);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 8, block_id / 8);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 512);
  {
    float *compute = output0;
    {
      float compute_local[1];

      compute_local[0] = 0.000000e+00f;
#pragma unroll
      for (int rc_outer = 0; rc_outer < 24; ++rc_outer) {
        __syncthreads();
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
          pad_temp_shared[(
              ((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] =
              input0[(
                  ((((rc_outer * 1024) + (((int)threadIdx.z) * 128)) +
                    ((((((int)threadIdx.x) * 2) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >>
                      3) *
                     64)) +
                   (((int)blockIdx.y) * 8)) +
                  (((((int)threadIdx.x) * 2) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) &
                   7))];
        }
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 2;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
          input1_shared[(
              ((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] =
              input1[(
                  ((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    (rc_outer * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)];
        }
        __syncthreads();
#pragma unroll
        for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
          compute_local[0] =
              (compute_local[0] +
               (pad_temp_shared[((rc_inner * 8) + ((int)threadIdx.x))] *
                input1_shared[((((int)threadIdx.z) * 16) + rc_inner)]));
        }
      }
      compute[((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 64)) +
                (((int)blockIdx.y) * 8)) +
               ((int)threadIdx.x))] = compute_local[0];
    }
  }
}
// Node name:	Slice_1344
// Description:	Slice
// Input:
//	- name: Pad_1339_0	type: float	shape: Shape{1, 384, 17, 17}
// Output:
//	- name: Slice_1344_0	type: float	shape: Shape{1, 384, 16, 16}
__device__ __forceinline__ void
Slice_float_float_cuda_Slice_1344_block_kernel(float *input0, float *output0,
                                               int thread_id, int block_id,
                                               char *shared_buffer) {
  if (thread_id >= 64) {
    return;
  }
  const dim3 blockDim(64, 1, 1);
  const dim3 gridDim(1536, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < 98304) {
    uint32_t input_strides[] = {110976, 289, 17, 1};
    uint32_t output_strides[] = {98304, 256, 16, 1};
    uint32_t lower_bounds[] = {0, 0, 1, 1};
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
// Node name:	Add_2511
// Description:	Add
// Input:
//	- name: Convolution_1341_0	type: float	shape: Shape{1, 128, 16,
//16}
//	- name: Constant_2928_0	type: float	shape: Shape{1, 128, 16, 16}
// Output:
//	- name: BatchNormInference_1345_0	type: float	shape: Shape{1,
//128, 16, 16}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2511_block_kernel(float *input0, float *input1,
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
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_Convolution_Slice_Add_120(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2) {

  __shared__ char shared_buffer[1024];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Add_float_float_float_cuda_Add_2511_block_kernel(
        input3, input4, output2, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_1343_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 64, shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 1663) {
    Slice_float_float_cuda_Slice_1344_block_kernel(
        input2, output1, threadIdx.x, blockIdx.x - 128, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_Convolution_Slice_Add_120_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_Convolution_Slice_Add_120<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    output0, output1, output2);
}
// Node name:	Constant_333
// Description:	Constant
// Input:
// Output:
//	- name: Constant_333_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_333(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_333_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_333_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
