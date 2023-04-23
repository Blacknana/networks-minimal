// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_76
// Description:	Constant
// Input:
// Output:
//	- name: Constant_76_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_76(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_76_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_76_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[256];
  bin_file.read(tmp_mem, 256);
  cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_403
// Description:	Constant
// Input:
// Output:
//	- name: Constant_403_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_403(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_403_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_403_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_99
// Description:	Constant
// Input:
// Output:
//	- name: Constant_99_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_99(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_99_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_99_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2473
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2473_0	type: float	shape: Shape{128, 384, 1, 1}
void Constant_float_cuda_Constant_2473(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2473_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2473_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[196608];
  bin_file.read(tmp_mem, 196608);
  cudaMemcpyAsync(output0, tmp_mem, 196608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2140
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2140_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2140(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2140_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2140_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2344
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2344_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2344(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2344_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2344_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_206
// Description:	Constant
// Input:
// Output:
//	- name: Constant_206_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_206(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_206_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_206_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2134
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2134_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2134(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2134_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2134_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2200
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2200_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2200(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2200_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2200_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2707
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2707_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2707(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2707_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2707_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3108
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3108_0	type: float	shape: Shape{1, 128, 16, 16}
void Constant_float_cuda_Constant_3108(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3108_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3108_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2683
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2683_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2683(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2683_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2683_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_33
// Description:	Constant
// Input:
// Output:
//	- name: Constant_33_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_33(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_33_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_33_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_747_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2182_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_748_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2185_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_753_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Convolution_755_0	type: float	shape: Shape{1, 32, 32,
//32}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_753<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_747_0, Constant_2182_0,
// Convolution_753_0);
// Convolution_float_float_float_cuda_Convolution_755<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_748_0, Constant_2185_0,
// Convolution_755_0); Deduped function map: <src_function_name :
// deduped_function_name> Convolution_float_float_float_cuda_Convolution_755 :
// Convolution_float_float_float_cuda_Convolution_753

// Node name:	Convolution_753
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_747_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2182_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_753_0	type: float	shape: Shape{1, 32, 32,
//32}
__device__ __forceinline__ static void
Convolution_float_float_float_cuda_Convolution_753_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_34(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  __shared__ char shared_buffer[3072];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_753_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_753_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64 + 0,
        shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_34_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_34<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2975_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_627_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Slice_582_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2821_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_629_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: BatchNormInference_561_0	type: float	shape: Shape{1,
//32, 32, 32}
// Output:
//	- name: Add_633_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_634_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_9<<<dim3(64, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_627_0, Constant_2975_0, Slice_582_0,
// Add_633_0); FusedKernel_float_float_float_float_cuda_Add_Add_10<<<dim3(64, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_629_0, Constant_2821_0,
// BatchNormInference_561_0, Add_634_0); Deduped function map:
// <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_10 :
// FusedKernel_float_float_float_float_cuda_Add_Add_9

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_627_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2975_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Slice_582_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_633_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2112<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_627_0, Constant_2975_0, BatchNormInference_631_0);
// Add_float_float_float_cuda_Add_633<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_631_0, Slice_582_0, Add_633_0);
__device__ __forceinline__ static void
FusedKernel_float_float_float_float_cuda_Add_Add_9_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_17(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    FusedKernel_float_float_float_float_cuda_Add_Add_9_block_kernel(
        input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    FusedKernel_float_float_float_float_cuda_Add_Add_9_block_kernel(
        input4, input3, input5, output1, threadIdx.x, blockIdx.x - 64 + 0,
        NULL);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_17_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_17<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, output0, output1);
}
