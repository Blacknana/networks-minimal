// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2828_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_687_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: BatchNormInference_624_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2775_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_689_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Slice_643_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_696_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_697_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_13<<<dim3(64, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_687_0, Constant_2828_0,
// BatchNormInference_624_0, Add_696_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_14<<<dim3(64, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_689_0, Constant_2775_0, Slice_643_0,
// Add_697_0); Deduped function map: <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_14 :
// FusedKernel_float_float_float_float_cuda_Add_Add_13

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_687_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2828_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_624_0	type: float	shape: Shape{1,
//32, 32, 32}
// Output:
//	- name: Add_696_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2148<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_687_0, Constant_2828_0, BatchNormInference_693_0);
// Add_float_float_float_cuda_Add_696<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_693_0, BatchNormInference_624_0, Add_696_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Add_13_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_26(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    FusedKernel_float_float_float_float_cuda_Add_Add_13_block_kernel(
        input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    FusedKernel_float_float_float_float_cuda_Add_Add_13_block_kernel(
        input4, input3, input5, output1, threadIdx.x, blockIdx.x - 64, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_26_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_26<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, output0, output1);
}
// Node name:	Constant_3152
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3152_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3152(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3152_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3152_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2542
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2542_0	type: float	shape: Shape{128, 512, 1, 1}
void Constant_float_cuda_Constant_2542(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2542_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2542_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[262144];
  bin_file.read(tmp_mem, 262144);
  cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_368
// Description:	Constant
// Input:
// Output:
//	- name: Constant_368_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_368(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_368_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_368_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_151
// Description:	Constant
// Input:
// Output:
//	- name: Constant_151_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_151(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_151_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_151_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2497
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2497_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2497(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2497_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2497_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_351
// Description:	Constant
// Input:
// Output:
//	- name: Constant_351_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_351(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_351_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_351_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2890
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2890_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2890(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2890_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2890_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2515
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2515_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2515(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2515_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2515_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2188
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2188_0	type: float	shape: Shape{32, 192, 1, 1}
void Constant_float_cuda_Constant_2188(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2188_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2188_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[24576];
  bin_file.read(tmp_mem, 24576);
  cudaMemcpyAsync(output0, tmp_mem, 24576, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2161
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2161_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2161(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2161_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2161_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_126
// Description:	Constant
// Input:
// Output:
//	- name: Constant_126_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_126(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_126_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_126_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_613_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_87_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_614_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_424_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Constant_2819_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_616_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2974_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_620_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Convolution_618_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2820_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: DepthwiseConv2dNative_621_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_622_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Add_630_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_624_0	type: float	shape: Shape{1,
//32, 32, 32}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_621<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_613_0, Constant_87_0,
// DepthwiseConv2dNative_621_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_622<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_614_0, Constant_424_0,
// DepthwiseConv2dNative_622_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_8<<<dim3(64, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_616_0, Constant_2819_0,
// Convolution_620_0, Constant_2974_0, Add_630_0);
// Add_float_float_float_cuda_Add_2106<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_618_0, Constant_2820_0, BatchNormInference_624_0); Deduped
// function map: <src_function_name : deduped_function_name>

// Node name:	DepthwiseConv2dNative_621
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_613_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_87_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_621_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_621_block_kernel(
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
// Node name:	DepthwiseConv2dNative_622
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_614_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_424_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_622_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_622_block_kernel(
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
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_616_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2819_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_620_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2974_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_630_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2103<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_616_0, Constant_2819_0, BatchNormInference_623_0);
// Add_float_float_float_cuda_Add_2109<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_620_0, Constant_2974_0, BatchNormInference_625_0);
// Add_float_float_float_cuda_Add_630<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_625_0, BatchNormInference_623_0, Add_630_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_8_block_kernel(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    int thread_id, int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(64, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = add(input2[tid], input3[tid]);
  float temp2 = add(temp1, temp0);
  output0[tid] = temp2;
}
// Node name:	Add_2106
// Description:	Add
// Input:
//	- name: Convolution_618_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2820_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_624_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2106_block_kernel(float *input0, float *input1,
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_15(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_8_block_kernel(
        input5, input4, input7, input6, output2, threadIdx.x, blockIdx.x - 0,
        NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Add_float_float_float_cuda_Add_2106_block_kernel(
        input8, input9, output3, threadIdx.x, blockIdx.x - 64, NULL);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 383) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_621_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 128, NULL);
  } else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 639) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_622_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 384, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_15_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_15<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, input8, input9,
                                    output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: Slice_582_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_581_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_585_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2092_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3000_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_583_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2086_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_2996_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_584_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2089_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_2998_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_588_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_586_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_607_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_605_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_606_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Relu_float_float_cuda_Relu_588<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Slice_582_0, Relu_588_0); Add_float_float_float_cuda_Add_586<<<dim3(64,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_581_0, AvgPool_581_0, Add_586_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2999<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_585_0, Constant_2092_0,
// Constant_3000_0, Relu_607_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2995<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_583_0, Constant_2086_0,
// Constant_2996_0, Relu_605_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2997<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_584_0, Constant_2089_0,
// Constant_2998_0, Relu_606_0); Deduped function map: <src_function_name :
// deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2995 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2999
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2997 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2999

// Node name:	Relu_588
// Description:	Relu
// Input:
//	- name: Slice_582_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_588_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void
Relu_float_float_cuda_Relu_588_block_kernel(float *input0, float *output0,
                                            int thread_id, int block_id,
                                            char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(64, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  output0[blockIdx.x * 512 + threadIdx.x] =
      relu(input0[blockIdx.x * 512 + threadIdx.x]);
}
// Node name:	Add_586
// Description:	Add
// Input:
//	- name: AvgPool_581_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_581_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_586_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void Add_float_float_float_cuda_Add_586_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
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
// Node name:	Matched_Pattern_2999
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_585_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2092_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3000_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_607_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2999_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Add_Matched_Pattern_Matched_Pattern_Matched_Pattern_12(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {

  __shared__ char shared_buffer[3072];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Relu_float_float_cuda_Relu_588_block_kernel(input0, output0, threadIdx.x,
                                                blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Add_float_float_float_cuda_Add_586_block_kernel(
        input1, input1, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2999_block_kernel(
        input2, input3, input4, output2, threadIdx.x, blockIdx.x - 128,
        shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2999_block_kernel(
        input5, input6, input7, output3, threadIdx.x, blockIdx.x - 192,
        shared_buffer);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2999_block_kernel(
        input8, input9, input10, output4, threadIdx.x, blockIdx.x - 256,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Add_Matched_Pattern_Matched_Pattern_Matched_Pattern_12_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Add_Matched_Pattern_Matched_Pattern_Matched_Pattern_12<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2936_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1466_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Slice_1421_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2762_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1468_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: BatchNormInference_1398_0	type: float	shape: Shape{1,
//128, 8, 8}
// Output:
//	- name: Add_1472_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Add_1473_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_55<<<dim3(16, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1466_0, Constant_2936_0, Slice_1421_0,
// Add_1472_0); FusedKernel_float_float_float_float_cuda_Add_Add_56<<<dim3(16,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1468_0, Constant_2762_0,
// BatchNormInference_1398_0, Add_1473_0); Deduped function map:
// <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_56 :
// FusedKernel_float_float_float_float_cuda_Add_Add_55

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1466_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2936_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Slice_1421_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1472_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2574<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1466_0, Constant_2936_0, BatchNormInference_1470_0);
// Add_float_float_float_cuda_Add_1472<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1470_0, Slice_1421_0, Add_1472_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Add_55_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_138(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    FusedKernel_float_float_float_float_cuda_Add_Add_55_block_kernel(
        input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31) {
    FusedKernel_float_float_float_float_cuda_Add_Add_55_block_kernel(
        input4, input3, input5, output1, threadIdx.x, blockIdx.x - 16, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_138_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_138<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, output0, output1);
}
