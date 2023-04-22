// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	Constant_2994
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2994_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2994(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2994_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2994_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2910
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2910_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2910(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2910_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2910_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3168
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3168_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3168(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3168_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3168_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3142
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3142_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3142(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3142_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3142_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2686
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2686_0	type: float	shape: Shape{128, 768, 1, 1}
void Constant_float_cuda_Constant_2686(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2686_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2686_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[393216];
  bin_file.read(tmp_mem, 393216);
  cudaMemcpyAsync(output0, tmp_mem, 393216, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2798
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2798_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2798(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2798_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2798_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2928
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2928_0	type: float	shape: Shape{1, 128, 16, 16}
void Constant_float_cuda_Constant_2928(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2928_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2928_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2647
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2647_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2647(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2647_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2647_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2374
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2374_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2374(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2374_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2374_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_133
// Description:	Constant
// Input:
// Output:
//	- name: Constant_133_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_133(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_133_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_133_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_424
// Description:	Constant
// Input:
// Output:
//	- name: Constant_424_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_424(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_424_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_424_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[3200];
  bin_file.read(tmp_mem, 3200);
  cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_289
// Description:	Constant
// Input:
// Output:
//	- name: Constant_289_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_289(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_289_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_289_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[3200];
  bin_file.read(tmp_mem, 3200);
  cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_889
// Description:	Constant
// Input:
// Output:
//	- name: Constant_889_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_889(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_889_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_889_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4];
  bin_file.read(tmp_mem, 4);
  cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Slice_895_0	type: float	shape: Shape{1, 64, 32, 32}
//	- name: AvgPool_896_0	type: float	shape: Shape{1, 192, 16, 16}
//	- name: Constant_1853_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Relu_899_0	type: float	shape: Shape{1, 64, 32, 32}
//	- name: Convolution_901_0	type: float	shape: Shape{1, 32, 16,
//16}
// Fused functions:
// Relu_float_float_cuda_Relu_899<<<dim3(128, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Slice_895_0, Relu_899_0);
// Convolution_float_float_float_cuda_Convolution_901<<<dim3(2, 8, 4), dim3(8,
// 2, 8), 0, 0>>>(AvgPool_896_0, Constant_1853_0, Convolution_901_0); Deduped
// function map: <src_function_name : deduped_function_name>

// Node name:	Relu_899
// Description:	Relu
// Input:
//	- name: Slice_895_0	type: float	shape: Shape{1, 64, 32, 32}
// Output:
//	- name: Relu_899_0	type: float	shape: Shape{1, 64, 32, 32}
__device__ __forceinline__ void
Relu_float_float_cuda_Relu_899_block_kernel(float *input0, float *output0,
                                            int thread_id, int block_id,
                                            char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(128, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  output0[blockIdx.x * 512 + threadIdx.x] =
      relu(input0[blockIdx.x * 512 + threadIdx.x]);
}
// Node name:	Convolution_901
// Description:	Convolution
// Input:
//	- name: AvgPool_896_0	type: float	shape: Shape{1, 192, 16, 16}
//	- name: Constant_1853_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_901_0	type: float	shape: Shape{1, 32, 16,
//16}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_901_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(8, 2, 8);
  const dim3 gridDim(2, 8, 4);
  const dim3 threadIdx(thread_id % 8, thread_id / 8 % 2, thread_id / 16);
  const dim3 blockIdx(block_id % 2, block_id / 2 % 8, block_id / 16);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 3072);
  {
    float *compute = output0;
    {
      float compute_local[1];

      compute_local[0] = 0.000000e+00f;
#pragma unroll
      for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
        __syncthreads();
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 6;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
          pad_temp_shared[(
              (((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 48)) +
               (((int)threadIdx.x) * 6)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] =
              input0[(
                  (((((((rc_outer * 12288) + (((int)threadIdx.z) * 1536)) +
                       (((int)threadIdx.y) * 768)) +
                      ((((((int)threadIdx.x) * 6) +
                         ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >>
                        4) *
                       256)) +
                     (((int)blockIdx.y) * 32)) +
                    (((((((int)threadIdx.x) * 6) +
                        ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) &
                       15) >>
                      3) *
                     16)) +
                   (((int)blockIdx.x) * 8)) +
                  (((((int)threadIdx.x) * 6) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) &
                   7))];
        }
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
          input1_shared[(
              (((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) +
               (((int)threadIdx.x) * 3)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] =
              input1[(
                  (((((((int)blockIdx.z) * 1536) + (((int)threadIdx.z) * 192)) +
                     (rc_outer * 48)) +
                    (((int)threadIdx.y) * 24)) +
                   (((int)threadIdx.x) * 3)) +
                  ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)];
        }
        __syncthreads();
#pragma unroll
        for (int rc_inner = 0; rc_inner < 48; ++rc_inner) {
          compute_local[0] =
              (compute_local[0] +
               (pad_temp_shared[(((rc_inner * 16) + (((int)threadIdx.y) * 8)) +
                                 ((int)threadIdx.x))] *
                input1_shared[((((int)threadIdx.z) * 48) + rc_inner)]));
        }
      }
      compute[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 (((int)threadIdx.y) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((int)threadIdx.x))] = compute_local[0];
    }
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_cuda_Relu_Convolution_56(
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {

  __shared__ char shared_buffer[4608];

  // if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127)
  if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 191) {
    // Relu_float_float_cuda_Relu_899_block_kernel(input0, output0, threadIdx.x,
    // blockIdx.x - 64, shared_buffer);
  }
  // else if((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
  else if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    // Convolution_float_float_float_cuda_Convolution_901_block_kernel(input1,
    // input2, output1, threadIdx.x, blockIdx.x - 0, shared_buffer);
  }
}

extern "C" __global__ void
Convolution_float_float_float_cuda_Convolution_901(float *input0, float *input1,
                                                   float *output0) {
  __shared__ char shared_buffer[4608];
  Convolution_float_float_float_cuda_Convolution_901_block_kernel(
      input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
}

extern void
Convolution_float_float_float_cuda_Convolution_901_block_kernel_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *output0) {
  Convolution_float_float_float_cuda_Convolution_901<<<grids, blocks, mem,
                                                       stream>>>(input0, input1,
                                                                 output0);
}
extern void
BlockFusionKernel_float_float_float_float_float_cuda_Relu_Convolution_56_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_cuda_Relu_Convolution_56<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1161_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2407_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3088_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1162_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2410_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3090_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1163_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2413_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3092_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: AvgPool_1157_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Slice_1159_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1182_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1183_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1184_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_1160_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1165_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3087<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_1161_0,
// Constant_2407_0, Constant_3088_0, Relu_1182_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3089<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_1162_0,
// Constant_2410_0, Constant_3090_0, Relu_1183_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3091<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_1163_0,
// Constant_2413_0, Constant_3092_0, Relu_1184_0);
// Add_float_float_float_cuda_Add_1160<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(AvgPool_1157_0, AvgPool_1157_0, Add_1160_0);
// Relu_float_float_cuda_Relu_1165<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Slice_1159_0, Relu_1165_0); Deduped function map: <src_function_name :
// deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3089 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3087
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3091 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3087

// Node name:	Matched_Pattern_3087
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1161_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2407_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3088_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1182_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3087_block_kernel(
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
// Node name:	Add_1160
// Description:	Add
// Input:
//	- name: AvgPool_1157_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: AvgPool_1157_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1160_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_1160_block_kernel(float *input0, float *input1,
                                                 float *output0, int thread_id,
                                                 int block_id,
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
// Node name:	Relu_1165
// Description:	Relu
// Input:
//	- name: Slice_1159_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1165_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Relu_float_float_cuda_Relu_1165_block_kernel(float *input0, float *output0,
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_Relu_95(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {

  __shared__ char shared_buffer[2048];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_1160_block_kernel(
        input9, input9, output3, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    Relu_float_float_cuda_Relu_1165_block_kernel(
        input10, output4, threadIdx.x, blockIdx.x - 32, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3087_block_kernel(
        input0, input1, input2, output0, threadIdx.x, blockIdx.x - 64,
        shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3087_block_kernel(
        input3, input4, input5, output1, threadIdx.x, blockIdx.x - 128,
        shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3087_block_kernel(
        input6, input7, input8, output2, threadIdx.x, blockIdx.x - 192,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_Relu_95_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_Relu_95<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4);
}
