// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_225
// Description:	Constant
// Input:
// Output:
//	- name: Constant_225_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_225(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_225_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_225_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2744
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2744_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2744(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2744_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2744_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_354
// Description:	Constant
// Input:
// Output:
//	- name: Constant_354_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_354(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_354_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_354_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2458
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2458_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2458(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2458_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2458_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2849
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2849_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2849(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2849_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2849_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2509
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2509_0	type: float	shape: Shape{128, 384, 1, 1}
void Constant_float_cuda_Constant_2509(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2509_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2509_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[196608];
  bin_file.read(tmp_mem, 196608);
  cudaMemcpyAsync(output0, tmp_mem, 196608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_377
// Description:	Constant
// Input:
// Output:
//	- name: Constant_377_0	type: float	shape: Shape{7, 7, 128, 1}
void Constant_float_cuda_Constant_377(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_377_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_377_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[25088];
  bin_file.read(tmp_mem, 25088);
  cudaMemcpyAsync(output0, tmp_mem, 25088, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2713
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2713_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2713(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2713_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2713_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2810
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2810_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2810(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2810_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2810_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2819
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2819_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2819(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2819_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2819_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3054
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3054_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3054(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3054_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3054_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3148
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3148_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3148(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3148_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3148_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	AvgPool_1337
// Description:	AvgPool
// Input:
//	- name: Relu_1336_0	type: float	shape: Shape{1, 384, 16, 16}
// Output:
//	- name: AvgPool_1337_0	type: float	shape: Shape{1, 384, 8, 8}
void AvgPool_float_float_cuda_lib_AvgPool_1337(cudnnHandle_t cudnn_handle,
                                               float *input0, float *output0) {
  cudnnTensorDescriptor_t input_desc;
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, 1, 384, 16, 16));
  cudnnTensorDescriptor_t output_desc;
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
  CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, 1, 384, 8, 8));
  cudnnPoolingDescriptor_t desc;
  cudnnCreatePoolingDescriptor(&desc);
  CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(
      desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
      CUDNN_NOT_PROPAGATE_NAN, 1, 1, 0, 0, 2, 2));
  const float alpha = 1.0;
  const float beta = 0.0;
  CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc,
                                      input0, &beta, output_desc, output0));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
  CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1449_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2563_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1451_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2569_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1450_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2566_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1434_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2557_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3136_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1435_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2560_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3138_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Convolution_1455_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Convolution_1459_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Convolution_1457_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Relu_1452_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1453_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1455<<<dim3(1, 4, 16), dim3(8,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_1449_0, Constant_2563_0,
// Convolution_1455_0);
// Convolution_float_float_float_cuda_Convolution_1459<<<dim3(1, 4, 16), dim3(8,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_1451_0, Constant_2569_0,
// Convolution_1459_0);
// Convolution_float_float_float_cuda_Convolution_1457<<<dim3(1, 4, 16), dim3(8,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_1450_0, Constant_2566_0,
// Convolution_1457_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3135<<<dim3(1,
// 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1434_0, Constant_2557_0,
// Constant_3136_0, Relu_1452_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3137<<<dim3(1,
// 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1435_0, Constant_2560_0,
// Constant_3138_0, Relu_1453_0); Deduped function map: <src_function_name :
// deduped_function_name> Convolution_float_float_float_cuda_Convolution_1459 :
// Convolution_float_float_float_cuda_Convolution_1455
// Convolution_float_float_float_cuda_Convolution_1457 :
// Convolution_float_float_float_cuda_Convolution_1455
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3137 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3135

// Node name:	Convolution_1455
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1449_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2563_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1455_0	type: float	shape: Shape{1, 128, 8,
// 8}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1455_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(8, 2, 8);
  const dim3 gridDim(1, 4, 16);
  const dim3 threadIdx(thread_id % 8, thread_id / 8 % 2, thread_id / 16);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 4, block_id / 4);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 1024);
  {
    float *compute = output0;
    {
      float compute_local[1];

      compute_local[0] = 0.000000e+00f;
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                   (((int)blockIdx.y) * 16)) +
                  (((int)threadIdx.x) * 2))];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                   (((int)threadIdx.y) * 8)) +
                  ((int)threadIdx.x))];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  1024)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  1025)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  16)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  2048)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  2049)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  32)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  3072)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  3073)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  48)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  4096)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  4097)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  64)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  5120)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  5121)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  80)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  6144)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  6145)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  96)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  7168)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  7169)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  112)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 64)) +
                 (((int)blockIdx.y) * 16)) +
                (((int)threadIdx.y) * 8)) +
               ((int)threadIdx.x))] = compute_local[0];
    }
  }
}
// Node name:	Matched_Pattern_3135
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1434_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2557_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3136_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1452_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3135_block_kernel(
    float *input0, float *input1, float *input2, float *output0, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(8, 2, 8);
  const dim3 gridDim(1, 4, 16);
  const dim3 threadIdx(thread_id % 8, thread_id / 8 % 2, thread_id / 16);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 4, block_id / 4);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 1024);
  {
    float *compute = output0;
    {
      float compute1[1];

      compute1[0] = 0.000000e+00f;
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                   (((int)blockIdx.y) * 16)) +
                  (((int)threadIdx.x) * 2))];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                   (((int)threadIdx.y) * 8)) +
                  ((int)threadIdx.x))];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  1024)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  1025)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  16)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  2048)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  2049)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  32)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  3072)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  3073)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  48)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  4096)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  4097)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  64)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  5120)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  5121)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  80)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  6144)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  6145)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  96)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  7168)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) +
                    (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.x) * 2)) +
                  7169)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     ((int)threadIdx.x))] =
          input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   ((int)threadIdx.x)) +
                  112)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 64)) +
                 (((int)blockIdx.y) * 16)) +
                (((int)threadIdx.y) * 8)) +
               ((int)threadIdx.x))] =
          max((compute1[0] +
               input2[((((int)blockIdx.z) * 8) + ((int)threadIdx.z))]),
              0.000000e+00f);
    }
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Matched_Pattern_Matched_Pattern_135(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {

  __shared__ char shared_buffer[1536];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_1455_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_1455_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    Convolution_float_float_float_cuda_Convolution_1455_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 128, shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3135_block_kernel(
        input6, input7, input8, output3, threadIdx.x, blockIdx.x - 192,
        shared_buffer);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3135_block_kernel(
        input9, input10, input11, output4, threadIdx.x, blockIdx.x - 256,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Matched_Pattern_Matched_Pattern_135_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Matched_Pattern_Matched_Pattern_135<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, input11, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_971_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_254_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Constant_126_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: AvgPool_972_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_908_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Relu_995_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_323_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Relu_997_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_290_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Relu_996_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_143_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_977_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: DepthwiseConv2dNative_978_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Add_979_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1000_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: DepthwiseConv2dNative_1002_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: DepthwiseConv2dNative_1001_0	type: float	shape: Shape{1,
// 64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_977<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_971_0, Constant_254_0,
// DepthwiseConv2dNative_977_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_978<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_971_0, Constant_126_0,
// DepthwiseConv2dNative_978_0); Add_float_float_float_cuda_Add_979<<<dim3(32,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_972_0, BatchNormInference_908_0,
// Add_979_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1000<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_995_0, Constant_323_0,
// DepthwiseConv2dNative_1000_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1002<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_997_0, Constant_290_0,
// DepthwiseConv2dNative_1002_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1001<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_996_0, Constant_143_0,
// DepthwiseConv2dNative_1001_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1000 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_977
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1002 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_978
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1001 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_977

// Node name:	DepthwiseConv2dNative_977
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_971_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_254_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_977_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_977_block_kernel(
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
  const int filter_height = 3;
  const int filter_width = 3;
  const int depth_multiplier = 1;
  const int stride = 1;
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
// Node name:	DepthwiseConv2dNative_978
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_971_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_126_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_978_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_978_block_kernel(
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
// Node name:	Add_979
// Description:	Add
// Input:
//	- name: AvgPool_972_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_908_0	type: float	shape: Shape{1,
// 64, 16, 16}
// Output:
//	- name: Add_979_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void Add_float_float_float_cuda_Add_979_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(32, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  output0[blockIdx.x * 512 + threadIdx.x] =
      add(input0[blockIdx.x * 512 + threadIdx.x],
          input1[blockIdx.x * 512 + threadIdx.x]);
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_69(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_979_block_kernel(
        input3, input4, output2, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 159) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_977_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 32, NULL);
  } else if ((int)blockIdx.x >= 160 && (int)blockIdx.x <= 287) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_978_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 160, NULL);
  } else if ((int)blockIdx.x >= 288 && (int)blockIdx.x <= 415) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_977_block_kernel(
        input5, input6, output3, threadIdx.x, blockIdx.x - 288, NULL);
  } else if ((int)blockIdx.x >= 416 && (int)blockIdx.x <= 543) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_978_block_kernel(
        input7, input8, output4, threadIdx.x, blockIdx.x - 416, NULL);
  } else if ((int)blockIdx.x >= 544 && (int)blockIdx.x <= 671) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_977_block_kernel(
        input9, input10, output5, threadIdx.x, blockIdx.x - 544, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_69_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_69<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4, output5);
}
