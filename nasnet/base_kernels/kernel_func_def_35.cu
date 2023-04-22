// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	Constant_213
// Description:	Constant
// Input:
// Output:
//	- name: Constant_213_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_213(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_213_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_213_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_109
// Description:	Constant
// Input:
// Output:
//	- name: Constant_109_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_109(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_109_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_109_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[3200];
  bin_file.read(tmp_mem, 3200);
  cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2167
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2167_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2167(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2167_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2167_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2953
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2953_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2953(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2953_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2953_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2764
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2764_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2764(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2764_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2764_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_45
// Description:	Constant
// Input:
// Output:
//	- name: Constant_45_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_45(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_45_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_45_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2284
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2284_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2284(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2284_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2284_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2440
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2440_0	type: float	shape: Shape{64, 384, 1, 1}
void Constant_float_cuda_Constant_2440(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2440_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2440_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[98304];
  bin_file.read(tmp_mem, 98304);
  cudaMemcpyAsync(output0, tmp_mem, 98304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_153
// Description:	Constant
// Input:
// Output:
//	- name: Constant_153_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_153(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_153_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_153_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2813
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2813_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2813(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2813_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2813_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2194
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2194_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2194(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2194_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2194_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3058
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3058_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3058(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3058_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3058_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_870_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2859_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_866_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2857_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_868_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2858_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_846_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Constant_2239_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3044_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_847_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Constant_2242_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3046_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_875_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: BatchNormInference_873_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: BatchNormInference_874_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Relu_863_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_864_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2253<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_870_0, Constant_2859_0, BatchNormInference_875_0);
// Add_float_float_float_cuda_Add_2247<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_866_0, Constant_2857_0, BatchNormInference_873_0);
// Add_float_float_float_cuda_Add_2250<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_868_0, Constant_2858_0, BatchNormInference_874_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3043<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_846_0, Constant_2239_0,
// Constant_3044_0, Relu_863_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3045<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_847_0, Constant_2242_0,
// Constant_3046_0, Relu_864_0); Deduped function map: <src_function_name :
// deduped_function_name> Add_float_float_float_cuda_Add_2247 :
// Add_float_float_float_cuda_Add_2253 Add_float_float_float_cuda_Add_2250 :
// Add_float_float_float_cuda_Add_2253
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3045 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3043

// Node name:	Add_2253
// Description:	Add
// Input:
//	- name: Convolution_870_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2859_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_875_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2253_block_kernel(float *input0, float *input1,
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
// Node name:	Matched_Pattern_3043
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_846_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Constant_2239_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3044_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_863_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3043_block_kernel(
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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Add_Add_Matched_Pattern_Matched_Pattern_50(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {

  __shared__ char shared_buffer[3072];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_2253_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    Add_float_float_float_cuda_Add_2253_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 32, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 95) {
    Add_float_float_float_cuda_Add_2253_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 64, shared_buffer);
  } else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 159) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3043_block_kernel(
        input6, input7, input8, output3, threadIdx.x, blockIdx.x - 96,
        shared_buffer);
  } else if ((int)blockIdx.x >= 160 && (int)blockIdx.x <= 223) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3043_block_kernel(
        input9, input10, input11, output4, threadIdx.x, blockIdx.x - 160,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Add_Add_Matched_Pattern_Matched_Pattern_50_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Add_Add_Matched_Pattern_Matched_Pattern_50<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, input11, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1384_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2527_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1385_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2530_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1386_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2533_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Add_1355_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: AvgPool_1394_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1400_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2536_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3128_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Convolution_1389_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Convolution_1391_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Convolution_1393_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Add_1399_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1405_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1389<<<dim3(1, 4, 16), dim3(8,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_1384_0, Constant_2527_0,
// Convolution_1389_0);
// Convolution_float_float_float_cuda_Convolution_1391<<<dim3(1, 4, 16), dim3(8,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_1385_0, Constant_2530_0,
// Convolution_1391_0);
// Convolution_float_float_float_cuda_Convolution_1393<<<dim3(1, 4, 16), dim3(8,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_1386_0, Constant_2533_0,
// Convolution_1393_0); Add_float_float_float_cuda_Add_1399<<<dim3(16, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Add_1355_0, AvgPool_1394_0, Add_1399_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3127<<<dim3(1,
// 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1400_0, Constant_2536_0,
// Constant_3128_0, Relu_1405_0); Deduped function map: <src_function_name :
// deduped_function_name> Convolution_float_float_float_cuda_Convolution_1391 :
// Convolution_float_float_float_cuda_Convolution_1389
// Convolution_float_float_float_cuda_Convolution_1393 :
// Convolution_float_float_float_cuda_Convolution_1389

// Node name:	Convolution_1389
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1384_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2527_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1389_0	type: float	shape: Shape{1, 128, 8,
// 8}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1389_block_kernel(
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
// Node name:	Add_1399
// Description:	Add
// Input:
//	- name: Add_1355_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: AvgPool_1394_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1399_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_1399_block_kernel(float *input0, float *input1,
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
__device__ __forceinline__ void
Fuse_AvgPool_1361_Add_1366(const float *input0, const float *input1,
                           float *output, int thread_id, int block_id,
                           char *shared_buffer) {

  if (thread_id >= 512) {
    return;
  }
  const int pooled_height = 8;
  const int pooled_width = 8;
  const int nthreads = 8192;
  int index = block_id * 512 + threadIdx.x;

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
        avgval = (input0[slice_offset + h * kWidth + w]) /
                     ((hend - hstart) * (wend - wstart)) +
                 avgval;
      }
    }

    // output
    output[index] = add(avgval, input1[index]);
  }
}
// Node name:	Matched_Pattern_3127
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1400_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2536_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3128_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1405_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3127_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Add_Matched_Pattern_128(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {

  __shared__ char shared_buffer[1536];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    Fuse_AvgPool_1361_Add_1366(input7, input6, output3, threadIdx.x,
                               blockIdx.x - 0, shared_buffer);
    // Add_float_float_float_cuda_Add_1399_block_kernel(input6, input7, output3,
    // threadIdx.x, blockIdx.x - 192 + 0, shared_buffer);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 79) {
    Convolution_float_float_float_cuda_Convolution_1389_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 16, shared_buffer);
  } else if ((int)blockIdx.x >= 80 && (int)blockIdx.x <= 143) {
    Convolution_float_float_float_cuda_Convolution_1389_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 80, shared_buffer);
  } else if ((int)blockIdx.x >= 144 && (int)blockIdx.x <= 207) {
    Convolution_float_float_float_cuda_Convolution_1389_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 144, shared_buffer);
  } else if ((int)blockIdx.x >= 208 && (int)blockIdx.x <= 271) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3127_block_kernel(
        input8, input9, input10, output4, threadIdx.x, blockIdx.x - 208,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Add_Matched_Pattern_128_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Add_Matched_Pattern_128<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4);
}
// Node name:	Convolution_1729
// Description:	Convolution
// Input:
//	- name: Relu_1727_0	type: float	shape: Shape{1, 768, 8, 8}
//	- name: Constant_2722_0	type: float	shape: Shape{128, 768, 1, 1}
// Output:
//	- name: Convolution_1729_0	type: float	shape: Shape{1, 128, 8,
// 8}
extern "C" __global__ void Convolution_float_float_float_cuda_Convolution_1729(
    float *input0, float *input1, float *output0) {
  __shared__ float pad_temp_shared[512];
  __shared__ float input1_shared[512];
  {
    float *compute = output0;
    {
      float compute_local[1];

      compute_local[0] = 0.000000e+00f;
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              (((((((int)threadIdx.z) * 128) +
                  ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                    2) *
                   64)) +
                 (((int)blockIdx.y) * 32)) +
                ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                 8)) +
               (((int)blockIdx.x) * 4)) +
              ((((int)threadIdx.x) & 1) * 2))]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[((((((((int)threadIdx.z) * 128) +
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
                       (((((int)threadIdx.x) * 2) + 1) & 3))]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              2048)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       2048)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              4096)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       4096)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              6144)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       6144)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              8192)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       8192)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              10240)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       10240)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              12288)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       12288)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              14336)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       14336)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              16384)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       16384)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              18432)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       18432)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              20480)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       20480)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              22528)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       22528)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              24576)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       24576)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              26624)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       26624)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              28672)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       28672)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              30720)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       30720)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              32768)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       32768)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              34816)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       34816)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              36864)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       36864)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              38912)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       38912)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              40960)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       40960)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              43008)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       43008)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              45056)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       45056)]);
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
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              47104)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
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
                       47104)]);
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
extern void Convolution_float_float_float_cuda_Convolution_1729_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *output0) {
  Convolution_float_float_float_cuda_Convolution_1729<<<grids, blocks, mem,
                                                        stream>>>(
      input0, input1, output0);
}
