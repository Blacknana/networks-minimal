// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_3110
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3110_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3110(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3110_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3110_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3172
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3172_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3172(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3172_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3172_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_331
// Description:	Constant
// Input:
// Output:
//	- name: Constant_331_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_331(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_331_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_331_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2887
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2887_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2887(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2887_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2887_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2757
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2757_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2757(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2757_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2757_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_360
// Description:	Constant
// Input:
// Output:
//	- name: Constant_360_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_360(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_360_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_360_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2762
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2762_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2762(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2762_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2762_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_350
// Description:	Constant
// Input:
// Output:
//	- name: Constant_350_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_350(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_350_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_350_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3036
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3036_0	type: float	shape: Shape{1, 64, 32, 32}
void Constant_float_cuda_Constant_3036(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3036_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3036_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[262144];
  bin_file.read(tmp_mem, 262144);
  cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2182
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2182_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2182(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2182_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2182_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_887_0	type: float	shape: Shape{1, 64, 32,
//32}
//	- name: Constant_2861_0	type: float	shape: Shape{1, 64, 32, 32}
//	- name: AvgPool_888_0	type: float	shape: Shape{1, 192, 16, 16}
//	- name: Constant_1852_0	type: float	shape: Shape{32, 192, 1, 1}
//	- name: Pad_890_0	type: float	shape: Shape{1, 192, 33, 33}
// Output:
//	- name: BatchNormInference_891_0	type: float	shape: Shape{1,
//64, 32, 32}
//	- name: Convolution_893_0	type: float	shape: Shape{1, 32, 16,
//16}
//	- name: Slice_894_0	type: float	shape: Shape{1, 192, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2262<<<dim3(128, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_887_0, Constant_2861_0, BatchNormInference_891_0);
// Convolution_float_float_float_cuda_Convolution_893<<<dim3(2, 8, 2), dim3(4,
// 2, 16), 0, 0>>>(AvgPool_888_0, Constant_1852_0, Convolution_893_0);
// Slice_float_float_cuda_Slice_894<<<dim3(3072, 1, 1), dim3(64, 1, 1), 0,
// 0>>>(Pad_890_0, Slice_894_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	Add_2262
// Description:	Add
// Input:
//	- name: Convolution_887_0	type: float	shape: Shape{1, 64, 32,
//32}
//	- name: Constant_2861_0	type: float	shape: Shape{1, 64, 32, 32}
// Output:
//	- name: BatchNormInference_891_0	type: float	shape: Shape{1,
//64, 32, 32}
__device__ __forceinline__ static void
Add_float_float_float_cuda_Add_2262_block_kernel(float *input0, float *input1,
                                                 float *output0, int thread_id,
                                                 int block_id,
                                                 char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(128, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  output0[blockIdx.x * 512 + threadIdx.x] =
      add(input0[blockIdx.x * 512 + threadIdx.x],
          input1[blockIdx.x * 512 + threadIdx.x]);
}
// Node name:	Convolution_893
// Description:	Convolution
// Input:
//	- name: AvgPool_888_0	type: float	shape: Shape{1, 192, 16, 16}
//	- name: Constant_1852_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_893_0	type: float	shape: Shape{1, 32, 16,
//16}
__device__ __forceinline__ static void
Convolution_float_float_float_cuda_Convolution_893_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(4, 2, 16);
  const dim3 gridDim(2, 8, 2);
  const dim3 threadIdx(thread_id % 4, thread_id / 4 % 2, thread_id / 8);
  const dim3 blockIdx(block_id % 2, block_id / 2 % 8, block_id / 16);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 1024);
  {
    float *compute = output0;
    {
      float compute_local[2];

      compute_local[0] = 0.000000e+00f;
      compute_local[1] = 0.000000e+00f;
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                    (((int)threadIdx.y) * 16)) +
                   (((int)blockIdx.x) * 8)) +
                  (((int)threadIdx.x) * 2))];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                   (((int)threadIdx.y) * 8)) +
                  (((int)threadIdx.x) * 2))];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             4)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             20)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             36)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             52)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             68)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             84)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             100)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             116)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             132)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             148)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             164)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             180)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             196)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             212)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             228)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             244)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  4096)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  4097)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  16)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  17)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             4)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             20)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             36)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             52)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             68)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             84)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             100)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             116)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             132)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             148)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             164)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             180)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             196)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             212)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             228)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             244)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  8192)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  8193)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  32)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  33)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             4)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             20)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             36)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             52)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             68)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             84)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             100)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             116)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             132)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             148)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             164)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             180)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             196)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             212)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             228)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             244)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  12288)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  12289)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  48)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  49)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             4)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             20)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             36)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             52)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             68)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             84)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             100)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             116)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             132)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             148)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             164)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             180)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             196)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             212)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             228)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             244)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  16384)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  16385)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  64)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  65)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             4)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             20)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             36)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             52)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             68)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             84)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             100)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             116)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             132)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             148)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             164)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             180)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             196)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             212)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             228)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             244)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  20480)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  20481)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  80)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  81)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             4)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             20)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             36)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             52)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             68)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             84)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             100)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             116)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             132)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             148)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             164)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             180)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             196)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             212)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             228)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             244)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  24576)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  24577)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  96)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  97)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             4)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             20)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             36)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             52)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             68)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             84)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             100)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             116)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             132)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             148)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             164)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             180)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             196)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             212)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             228)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             244)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  28672)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  28673)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  112)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  113)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             4)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             20)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             36)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             52)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             68)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             84)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             100)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             116)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             132)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             148)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             164)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             180)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             196)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             212)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             228)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             244)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  32768)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  32769)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  128)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  129)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             4)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             20)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             36)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             52)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             68)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             84)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             100)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             116)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             132)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             148)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             164)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             180)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             196)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             212)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             228)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             244)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  36864)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  36865)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  144)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  145)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             4)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             20)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             36)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             52)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             68)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             84)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             100)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             116)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             132)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             148)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             164)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             180)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             196)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             212)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             228)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             244)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  40960)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  40961)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  160)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  161)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             4)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             20)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             36)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             52)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             68)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             84)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             100)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             116)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             132)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             148)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             164)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             180)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             196)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             212)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             228)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             244)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  45056)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) +
                     (((int)threadIdx.y) * 16)) +
                    (((int)blockIdx.x) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  45057)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  176)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 192)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  177)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             4)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             20)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             36)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             52)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             68)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             84)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             100)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             116)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             132)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             148)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             164)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             180)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             196)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             212)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             228)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             244)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 (((int)threadIdx.y) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((int)threadIdx.x))] = compute_local[0];
      compute[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                   (((int)blockIdx.y) * 32)) +
                  (((int)threadIdx.y) * 16)) +
                 (((int)blockIdx.x) * 8)) +
                ((int)threadIdx.x)) +
               4)] = compute_local[1];
    }
  }
}
// Node name:	Slice_894
// Description:	Slice
// Input:
//	- name: Pad_890_0	type: float	shape: Shape{1, 192, 33, 33}
// Output:
//	- name: Slice_894_0	type: float	shape: Shape{1, 192, 32, 32}
__device__ __forceinline__ static void
Slice_float_float_cuda_Slice_894_block_kernel(float *input0, float *output0,
                                              int thread_id, int block_id,
                                              char *shared_buffer) {
  if (thread_id >= 64) {
    return;
  }
  const dim3 blockDim(64, 1, 1);
  const dim3 gridDim(3072, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < 196608) {
    uint32_t input_strides[] = {209088, 1089, 33, 1};
    uint32_t output_strides[] = {196608, 1024, 32, 1};
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
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_Add_Convolution_Slice_55(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2) {
  __shared__ char shared_buffer[2048];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127) {
    Add_float_float_float_cuda_Add_2262_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 159) {
    Convolution_float_float_float_cuda_Convolution_893_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 128 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 160 && (int)blockIdx.x <= 3231) {
    Slice_float_float_cuda_Slice_894_block_kernel(
        input4, output2, threadIdx.x, blockIdx.x - 160 + 0, shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_Add_Convolution_Slice_55_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_Add_Convolution_Slice_55<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    output0, output1, output2);
}
// Node name:	Constant_3074
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3074_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3074(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3074_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3074_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1420_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_283_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Constant_213_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Constant_274_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: BatchNormInference_1418_0	type: float	shape: Shape{1,
//128, 8, 8}
// Output:
//	- name: DepthwiseConv2dNative_1423_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: DepthwiseConv2dNative_1425_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: DepthwiseConv2dNative_1424_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Slice_1421_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1423<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1420_0, Constant_283_0,
// DepthwiseConv2dNative_1423_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1425<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1420_0, Constant_213_0,
// DepthwiseConv2dNative_1425_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1424<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1420_0, Constant_274_0,
// DepthwiseConv2dNative_1424_0); Slice_float_float_cuda_Slice_1421<<<dim3(128,
// 1, 1), dim3(64, 1, 1), 0, 0>>>(BatchNormInference_1418_0, Slice_1421_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1424 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1423

// Node name:	DepthwiseConv2dNative_1423
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1420_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_283_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1423_0	type: float	shape: Shape{1,
//128, 8, 8}
__device__ __forceinline__ static void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1423_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(128, 1, 1);
  const dim3 gridDim(64, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);

  typedef float S;
  float *input = input0;
  float *filter = input1;
  float *output = output0;

  const int in_height = 8;
  const int in_width = 8;
  const int in_depth = 128;
  const int filter_height = 3;
  const int filter_width = 3;
  const int depth_multiplier = 1;
  const int stride = 1;
  const int pad_height = 1;
  const int pad_width = 1;
  const int out_height = 8;
  const int out_width = 8;
  const int out_depth = 128;
  const int num_outputs = 8192;

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
// Node name:	DepthwiseConv2dNative_1425
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1420_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_213_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1425_0	type: float	shape: Shape{1,
//128, 8, 8}
__device__ __forceinline__ static void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1425_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(128, 1, 1);
  const dim3 gridDim(64, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);

  typedef float S;
  float *input = input0;
  float *filter = input1;
  float *output = output0;

  const int in_height = 8;
  const int in_width = 8;
  const int in_depth = 128;
  const int filter_height = 5;
  const int filter_width = 5;
  const int depth_multiplier = 1;
  const int stride = 1;
  const int pad_height = 2;
  const int pad_width = 2;
  const int out_height = 8;
  const int out_width = 8;
  const int out_depth = 128;
  const int num_outputs = 8192;

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
// Node name:	Slice_1421
// Description:	Slice
// Input:
//	- name: BatchNormInference_1418_0	type: float	shape: Shape{1,
//128, 8, 8}
// Output:
//	- name: Slice_1421_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ static void
Slice_float_float_cuda_Slice_1421_block_kernel(float *input0, float *output0,
                                               int thread_id, int block_id,
                                               char *shared_buffer) {
  if (thread_id >= 64) {
    return;
  }
  const dim3 blockDim(64, 1, 1);
  const dim3 gridDim(128, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < 8192) {
    uint32_t input_strides[] = {8192, 64, 8, 1};
    uint32_t output_strides[] = {8192, 64, 8, 1};
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_132(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1423_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1425_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 64 + 0, NULL);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1423_block_kernel(
        input0, input3, output2, threadIdx.x, blockIdx.x - 128 + 0, NULL);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 319) {
    Slice_float_float_cuda_Slice_1421_block_kernel(input4, output3, threadIdx.x,
                                                   blockIdx.x - 192 + 0, NULL);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_132_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_132<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1601_0	type: float	shape: Shape{1, 768, 8, 8}
//	- name: Constant_2650_0	type: float	shape: Shape{128, 768, 1, 1}
//	- name: Constant_2653_0	type: float	shape: Shape{128, 768, 1, 1}
// Output:
//	- name: Convolution_1603_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Convolution_1605_0	type: float	shape: Shape{1, 128, 8,
//8}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1603<<<dim3(2, 2, 8), dim3(4,
// 4, 16), 0, 0>>>(Relu_1601_0, Constant_2650_0, Convolution_1603_0);
// Convolution_float_float_float_cuda_Convolution_1605<<<dim3(2, 2, 8), dim3(4,
// 4, 16), 0, 0>>>(Relu_1601_0, Constant_2653_0, Convolution_1605_0); Deduped
// function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1605 :
// Convolution_float_float_float_cuda_Convolution_1603

// Node name:	Convolution_1603
// Description:	Convolution
// Input:
//	- name: Relu_1601_0	type: float	shape: Shape{1, 768, 8, 8}
//	- name: Constant_2650_0	type: float	shape: Shape{128, 768, 1, 1}
// Output:
//	- name: Convolution_1603_0	type: float	shape: Shape{1, 128, 8,
//8}
__device__ __forceinline__ static void
Convolution_float_float_float_cuda_Convolution_1603_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(4, 4, 16);
  const dim3 gridDim(2, 2, 8);
  const dim3 threadIdx(thread_id % 4, thread_id / 4 % 4, thread_id / 16);
  const dim3 blockIdx(block_id % 2, block_id / 2 % 2, block_id / 4);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 2048);
  {
    float *compute = output0;
    {
      float compute_local[1];

      compute_local[0] = 0.000000e+00f;
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] = input0
          [((((((((int)threadIdx.z) * 128) +
                ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) *
                 64)) +
               (((int)blockIdx.y) * 32)) +
              ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
               8)) +
             (((int)blockIdx.x) * 4)) +
            ((((int)threadIdx.x) & 1) * 2))];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[((((((((int)threadIdx.z) * 128) +
                      ((((((int)threadIdx.y) * 2) +
                         (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                        2) *
                       64)) +
                     (((int)blockIdx.y) * 32)) +
                    ((((((int)threadIdx.y) * 2) +
                       (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                      3) *
                     8)) +
                   (((int)blockIdx.x) * 4)) +
                  (((((int)threadIdx.x) * 2) + 1) & 3))];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                   (((int)threadIdx.y) * 8)) +
                  (((int)threadIdx.x) * 2))];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              2048)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  2048)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  32)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  33)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              4096)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  4096)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  64)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  65)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              6144)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  6144)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  96)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  97)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              8192)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  8192)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  128)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  129)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              10240)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  10240)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  160)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  161)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              12288)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  12288)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  192)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  193)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              14336)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  14336)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  224)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  225)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              16384)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  16384)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  256)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  257)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              18432)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  18432)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  288)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  289)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              20480)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  20480)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  320)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  321)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              22528)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  22528)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  352)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  353)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              24576)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  24576)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  384)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  385)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              26624)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  26624)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  416)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  417)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              28672)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  28672)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  448)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  449)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              30720)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  30720)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  480)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  481)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              32768)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  32768)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  512)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  513)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              34816)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  34816)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  544)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  545)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              36864)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  36864)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  576)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  577)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              38912)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  38912)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  608)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  609)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              40960)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  40960)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  640)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  641)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              43008)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  43008)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  672)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  673)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              45056)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  45056)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  704)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  705)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              47104)];
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((((int)threadIdx.z) * 128) +
                       ((((((int)threadIdx.y) * 2) +
                          (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                         2) *
                        64)) +
                      (((int)blockIdx.y) * 32)) +
                     ((((((int)threadIdx.y) * 2) +
                        (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                       3) *
                      8)) +
                    (((int)blockIdx.x) * 4)) +
                   (((((int)threadIdx.x) * 2) + 1) & 3)) +
                  47104)];
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  736)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  737)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute[((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                  (((int)blockIdx.y) * 32)) +
                 (((int)threadIdx.y) * 8)) +
                (((int)blockIdx.x) * 4)) +
               ((int)threadIdx.x))] = compute_local[0];
    }
  }
}
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_157(
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  __shared__ char shared_buffer[4096];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Convolution_float_float_float_cuda_Convolution_1603_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_1603_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 32 + 0,
        shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_157_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_157<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1063_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2356_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_1061_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2350_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_1062_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2353_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_1046_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2344_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3074_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1047_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2347_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3076_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Convolution_1071_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Convolution_1067_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Convolution_1069_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Relu_1064_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1065_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1071<<<dim3(1, 16, 4), dim3(8,
// 1, 16), 0, 0>>>(DepthwiseConv2dNative_1063_0, Constant_2356_0,
// Convolution_1071_0);
// Convolution_float_float_float_cuda_Convolution_1067<<<dim3(1, 16, 4), dim3(8,
// 1, 16), 0, 0>>>(DepthwiseConv2dNative_1061_0, Constant_2350_0,
// Convolution_1067_0);
// Convolution_float_float_float_cuda_Convolution_1069<<<dim3(1, 16, 4), dim3(8,
// 1, 16), 0, 0>>>(DepthwiseConv2dNative_1062_0, Constant_2353_0,
// Convolution_1069_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3073<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_1046_0,
// Constant_2344_0, Constant_3074_0, Relu_1064_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3075<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_1047_0,
// Constant_2347_0, Constant_3076_0, Relu_1065_0); Deduped function map:
// <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1067 :
// Convolution_float_float_float_cuda_Convolution_1071
// Convolution_float_float_float_cuda_Convolution_1069 :
// Convolution_float_float_float_cuda_Convolution_1071
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3075 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3073

// Node name:	Convolution_1071
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1063_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2356_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_1071_0	type: float	shape: Shape{1, 64, 16,
//16}
__device__ __forceinline__ static void
Convolution_float_float_float_cuda_Convolution_1071_block_kernel(
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
// Node name:	Matched_Pattern_3073
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1046_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2344_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3074_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1064_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ static void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3073_block_kernel(
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
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Matched_Pattern_Matched_Pattern_79(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {
  __shared__ char shared_buffer[2048];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_1071_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_1071_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    Convolution_float_float_float_cuda_Convolution_1071_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 128 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3073_block_kernel(
        input6, input7, input8, output3, threadIdx.x, blockIdx.x - 192 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3073_block_kernel(
        input9, input10, input11, output4, threadIdx.x, blockIdx.x - 256 + 0,
        shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Matched_Pattern_Matched_Pattern_79_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Matched_Pattern_Matched_Pattern_79<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, input11, output0, output1, output2, output3, output4);
}
