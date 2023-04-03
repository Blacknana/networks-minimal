// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_2812
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2812_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2812(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2812_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2812_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2398
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2398_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2398(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2398_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2398_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2653
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2653_0	type: float	shape: Shape{128, 768, 1, 1}
void Constant_float_cuda_Constant_2653(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2653_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2653_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[393216];
  bin_file.read(tmp_mem, 393216);
  cudaMemcpyAsync(output0, tmp_mem, 393216, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2068
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2068_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2068(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2068_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2068_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2038
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2038_0	type: float	shape: Shape{32, 96, 1, 1}
void Constant_float_cuda_Constant_2038(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2038_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2038_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12288];
  bin_file.read(tmp_mem, 12288);
  cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2488
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2488_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2488(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2488_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2488_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_175
// Description:	Constant
// Input:
// Output:
//	- name: Constant_175_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_175(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_175_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_175_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2846
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2846_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2846(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2846_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2846_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2822
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2822_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2822(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2822_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2822_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2873
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2873_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2873(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2873_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2873_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3182
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3182_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3182(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3182_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3182_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_269
// Description:	Constant
// Input:
// Output:
//	- name: Constant_269_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_269(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_269_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_269_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2940_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1529_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: BatchNormInference_1462_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2941_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1531_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Slice_1484_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1535_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Add_1536_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_59<<<dim3(16, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1529_0, Constant_2940_0,
// BatchNormInference_1462_0, Add_1535_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_60<<<dim3(16, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1531_0, Constant_2941_0, Slice_1484_0,
// Add_1536_0); Deduped function map: <src_function_name :
// deduped_function_name> FusedKernel_float_float_float_float_cuda_Add_Add_60 :
// FusedKernel_float_float_float_float_cuda_Add_Add_59

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1529_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2940_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1462_0	type: float	shape: Shape{1,
//128, 8, 8}
// Output:
//	- name: Add_1535_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2610<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1529_0, Constant_2940_0, BatchNormInference_1533_0);
// Add_float_float_float_cuda_Add_1535<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1533_0, BatchNormInference_1462_0, Add_1535_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Add_59_block_kernel(
    float *input0, float *input1, float *input2, float *output0, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(16, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = add(temp0, input2[tid]);
  output0[tid] = temp1;
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_147(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    FusedKernel_float_float_float_float_cuda_Add_Add_59_block_kernel(
        input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31) {
    FusedKernel_float_float_float_float_cuda_Add_Add_59_block_kernel(
        input4, input3, input5, output1, threadIdx.x, blockIdx.x - 16, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_147_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_147<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2757_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1589_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Slice_1545_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2952_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1591_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: BatchNormInference_1525_0	type: float	shape: Shape{1,
//128, 8, 8}
// Output:
//	- name: Add_1598_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Add_1599_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_63<<<dim3(16, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1589_0, Constant_2757_0, Slice_1545_0,
// Add_1598_0); FusedKernel_float_float_float_float_cuda_Add_Add_64<<<dim3(16,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1591_0, Constant_2952_0,
// BatchNormInference_1525_0, Add_1599_0); Deduped function map:
// <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_64 :
// FusedKernel_float_float_float_float_cuda_Add_Add_63

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1589_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2757_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Slice_1545_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1598_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2646<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1589_0, Constant_2757_0, BatchNormInference_1595_0);
// Add_float_float_float_cuda_Add_1598<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1595_0, Slice_1545_0, Add_1598_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Add_63_block_kernel(
    float *input0, float *input1, float *input2, float *output0, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(16, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = add(temp0, input2[tid]);
  output0[tid] = temp1;
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_156(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    FusedKernel_float_float_float_float_cuda_Add_Add_63_block_kernel(
        input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31) {
    FusedKernel_float_float_float_float_cuda_Add_Add_63_block_kernel(
        input4, input3, input5, output1, threadIdx.x, blockIdx.x - 16, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_156_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_156<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: BatchNormInference_579_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Relu_580_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_140_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Constant_18_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Constant_144_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: Slice_582_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_585_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_583_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_584_0	type: float	shape: Shape{1,
//32, 32, 32}
// Fused functions:
// Slice_float_float_cuda_Slice_582<<<dim3(512, 1, 1), dim3(64, 1, 1), 0,
// 0>>>(BatchNormInference_579_0, Slice_582_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_585<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_580_0, Constant_140_0,
// DepthwiseConv2dNative_585_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_583<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_580_0, Constant_18_0,
// DepthwiseConv2dNative_583_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_584<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_580_0, Constant_144_0,
// DepthwiseConv2dNative_584_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_584 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_583

// Node name:	Slice_582
// Description:	Slice
// Input:
//	- name: BatchNormInference_579_0	type: float	shape: Shape{1,
//32, 32, 32}
// Output:
//	- name: Slice_582_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void
Slice_float_float_cuda_Slice_582_block_kernel(float *input0, float *output0,
                                              int thread_id, int block_id,
                                              char *shared_buffer) {
  if (thread_id >= 64) {
    return;
  }
  const dim3 blockDim(64, 1, 1);
  const dim3 gridDim(512, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < 32768) {
    uint32_t input_idx = tid;
    /*
    uint32_t input_strides[] = {32768, 1024, 32, 1};
    uint32_t output_strides[] = {32768, 1024, 32, 1};
    uint32_t lower_bounds[] = {0, 0, 0, 0};
    uint32_t slice_strides[] = {1, 1, 1, 1};

    uint32_t output_idx = tid;
    input_idx += (((output_idx / output_strides[0]) * slice_strides[0]) +
    lower_bounds[0]) * input_strides[0]; output_idx %= output_strides[0];
    input_idx += (((output_idx / output_strides[1]) * slice_strides[1]) +
    lower_bounds[1]) * input_strides[1]; output_idx %= output_strides[1];
    input_idx += (((output_idx / output_strides[2]) * slice_strides[2]) +
    lower_bounds[2]) * input_strides[2]; output_idx %= output_strides[2];
    input_idx += (((output_idx / output_strides[3]) * slice_strides[3]) +
    lower_bounds[3]) * input_strides[3];
    */
    output0[tid] = input0[input_idx];
  }
}
// Node name:	DepthwiseConv2dNative_585
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_580_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_140_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_585_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_585_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(128, 1, 1);
  const dim3 gridDim(256, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);

  typedef float S;
  float *input = input0;
  float *filter = input1;
  float *output = output0;

  const int in_height = 32;
  const int in_width = 32;
  const int in_depth = 32;
  const int filter_height = 5;
  const int filter_width = 5;
  const int depth_multiplier = 1;
  const int stride = 1;
  const int pad_height = 2;
  const int pad_width = 2;
  const int out_height = 32;
  const int out_width = 32;
  const int out_depth = 32;
  const int num_outputs = 32768;

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
__device__ __forceinline__ static void
fused2_DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_583_block_kernel(
    float *input0, float *input1, float *input2, float *output0, float *output1,
    int thread_id, int block_id, char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(128, 1, 1);
  const dim3 gridDim(256, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);

  typedef float S;
  float *input = input0;
  float *filter = input1;
  float *output = output0;

  const int in_height = 32;
  const int in_width = 32;
  const int in_depth = 32;
  const int filter_height = 3;
  const int filter_width = 3;
  const int depth_multiplier = 1;
  const int stride = 1;
  const int pad_height = 1;
  const int pad_width = 1;
  const int out_height = 32;
  const int out_width = 32;
  const int out_depth = 32;
  const int num_outputs = 32768;

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
// Node name:	DepthwiseConv2dNative_583
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_580_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_18_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_583_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_583_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(128, 1, 1);
  const dim3 gridDim(256, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);

  typedef float S;
  float *input = input0;
  float *filter = input1;
  float *output = output0;

  const int in_height = 32;
  const int in_width = 32;
  const int in_depth = 32;
  const int filter_height = 3;
  const int filter_width = 3;
  const int depth_multiplier = 1;
  const int stride = 1;
  const int pad_height = 1;
  const int pad_width = 1;
  const int out_height = 32;
  const int out_width = 32;
  const int out_depth = 32;
  const int num_outputs = 32768;

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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_Slice_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_11(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 255) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_585_block_kernel(
        input1, input2, output1, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 511) {
    fused2_DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_583_block_kernel(
        input1, input3, input4, output2, output3, threadIdx.x, blockIdx.x - 256,
        NULL);
  }
  // else if((int)blockIdx.x >= 512 && (int)blockIdx.x <= 767)
  //{
  // DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_583_block_kernel(input1,
  // input4, output3, threadIdx.x, blockIdx.x - 512, NULL);
  //}
  else if ((int)blockIdx.x >= 768 - 256 && (int)blockIdx.x <= 1279 - 256) {
    Slice_float_float_cuda_Slice_582_block_kernel(input0, output0, threadIdx.x,
                                                  blockIdx.x - 768 + 256, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_Slice_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_11_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_Slice_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_11<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    output0, output1, output2, output3);
}
