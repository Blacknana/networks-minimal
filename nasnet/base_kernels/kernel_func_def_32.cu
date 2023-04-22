// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	Constant_3046
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3046_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_3046(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3046_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3046_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2756
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2756_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2756(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2756_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2756_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2950
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2950_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2950(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2950_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2950_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_316
// Description:	Constant
// Input:
// Output:
//	- name: Constant_316_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_316(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_316_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_316_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[512];
  bin_file.read(tmp_mem, 512);
  cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_264
// Description:	Constant
// Input:
// Output:
//	- name: Constant_264_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_264(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_264_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_264_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_410
// Description:	Constant
// Input:
// Output:
//	- name: Constant_410_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_410(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_410_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_410_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2179
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2179_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2179(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2179_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2179_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3136
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3136_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3136(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3136_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3136_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3162
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3162_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3162(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3162_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3162_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2782_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1026_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Convolution_1028_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2805_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1029_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Relu_1031_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1030_0	type: float	shape: Shape{1,
// 64, 16, 16}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Relu_32<<<dim3(32, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1026_0, Constant_2782_0, Relu_1031_0,
// BatchNormInference_1029_0); Add_float_float_float_cuda_Add_2334<<<dim3(32, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1028_0, Constant_2805_0,
// BatchNormInference_1030_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1026_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2782_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1031_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1029_0	type: float	shape: Shape{1,
// 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2331<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1026_0, Constant_2782_0, BatchNormInference_1029_0);
// Relu_float_float_cuda_Relu_1031<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1029_0, Relu_1031_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Relu_32_block_kernel(
    float *input0, float *input1, float *output0, float *output1, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(32, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = relu(temp0);
  output1[tid] = temp0;
  output0[tid] = temp1;
}
// Node name:	Add_2334
// Description:	Add
// Input:
//	- name: Convolution_1028_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2805_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1030_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2334_block_kernel(float *input0, float *input1,
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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_75(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    FusedKernel_float_float_float_float_cuda_Add_Relu_32_block_kernel(
        input1, input0, output1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    Add_float_float_float_cuda_Add_2334_block_kernel(
        input2, input3, output2, threadIdx.x, blockIdx.x - 32, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_75_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {
  BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_75<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1, output2);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_707_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_437_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Constant_109_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Constant_426_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: BatchNormInference_705_0	type: float	shape: Shape{1,
// 32, 32, 32}
// Output:
//	- name: DepthwiseConv2dNative_710_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: DepthwiseConv2dNative_712_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: DepthwiseConv2dNative_711_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Slice_708_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_710<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_707_0, Constant_437_0,
// DepthwiseConv2dNative_710_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_712<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_707_0, Constant_109_0,
// DepthwiseConv2dNative_712_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_711<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_707_0, Constant_426_0,
// DepthwiseConv2dNative_711_0); Slice_float_float_cuda_Slice_708<<<dim3(512, 1,
// 1), dim3(64, 1, 1), 0, 0>>>(BatchNormInference_705_0, Slice_708_0); Deduped
// function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_711 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_710

// Node name:	DepthwiseConv2dNative_710
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_707_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_437_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_710_0	type: float	shape: Shape{1,
// 32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_710_block_kernel(
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
__device__ __forceinline__ static void
fused2_DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_710_block_kernel(
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
// Node name:	DepthwiseConv2dNative_712
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_707_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_109_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_712_0	type: float	shape: Shape{1,
// 32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_712_block_kernel(
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
// Node name:	Slice_708
// Description:	Slice
// Input:
//	- name: BatchNormInference_705_0	type: float	shape: Shape{1,
// 32, 32, 32}
// Output:
//	- name: Slice_708_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void
Slice_float_float_cuda_Slice_708_block_kernel(float *input0, float *output0,
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
    uint32_t input_strides[] = {32768, 1024, 32, 1};
    uint32_t output_strides[] = {32768, 1024, 32, 1};
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_29(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 255) {
    fused2_DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_710_block_kernel(
        input0, input1, input3, output0, output2, threadIdx.x, blockIdx.x - 0,
        NULL);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 511) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_712_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 256, NULL);
  }
  // else if((int)blockIdx.x >= 512 && (int)blockIdx.x <= 767)
  //{
  //    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_710_block_kernel(input0,
  //    input3, output2, threadIdx.x, blockIdx.x - 512, NULL);
  //}
  else if ((int)blockIdx.x >= 768 - 256 && (int)blockIdx.x <= 1279 - 256) {
    Slice_float_float_cuda_Slice_708_block_kernel(input4, output3, threadIdx.x,
                                                  blockIdx.x - 768 + 256, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_29_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_29<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_524_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Constant_2062_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_523_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Constant_2059_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Slice_500_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_507_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Constant_2047_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_2990_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_506_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Constant_2044_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_2988_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_499_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_505_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Constant_2041_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_2986_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_513_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Constant_2050_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_537_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Convolution_535_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Relu_510_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_541_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_540_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_508_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_539_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_526_0	type: float	shape: Shape{1, 32, 32,
// 32}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_537<<<dim3(1, 32, 2), dim3(16,
// 1, 4), 0, 0>>>(DepthwiseConv2dNative_524_0, Constant_2062_0,
// Convolution_537_0);
// Convolution_float_float_float_cuda_Convolution_535<<<dim3(1, 32, 2), dim3(16,
// 1, 4), 0, 0>>>(DepthwiseConv2dNative_523_0, Constant_2059_0,
// Convolution_535_0); Relu_float_float_cuda_Relu_510<<<dim3(64, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Slice_500_0, Relu_510_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989<<<dim3(1,
// 32, 2), dim3(16, 1, 4), 0, 0>>>(DepthwiseConv2dNative_507_0, Constant_2047_0,
// Constant_2990_0, Relu_541_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2987<<<dim3(1,
// 32, 2), dim3(16, 1, 4), 0, 0>>>(DepthwiseConv2dNative_506_0, Constant_2044_0,
// Constant_2988_0, Relu_540_0); Add_float_float_float_cuda_Add_508<<<dim3(64,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_499_0, AvgPool_499_0, Add_508_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2985<<<dim3(1,
// 32, 2), dim3(16, 1, 4), 0, 0>>>(DepthwiseConv2dNative_505_0, Constant_2041_0,
// Constant_2986_0, Relu_539_0);
// Convolution_float_float_float_cuda_Convolution_526<<<dim3(1, 32, 2), dim3(16,
// 1, 4), 0, 0>>>(DepthwiseConv2dNative_513_0, Constant_2050_0,
// Convolution_526_0); Deduped function map: <src_function_name :
// deduped_function_name> Convolution_float_float_float_cuda_Convolution_535 :
// Convolution_float_float_float_cuda_Convolution_537
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2987 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2985 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989
// Convolution_float_float_float_cuda_Convolution_526 :
// Convolution_float_float_float_cuda_Convolution_537

// Node name:	Convolution_537
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_524_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Constant_2062_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_537_0	type: float	shape: Shape{1, 32, 32,
// 32}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_537_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 64) {
    return;
  }
  const dim3 blockDim(16, 1, 4);
  const dim3 gridDim(1, 32, 2);
  const dim3 threadIdx(thread_id % 16, 0, thread_id / 16);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 32, block_id / 32);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 2048);
  {
    float *compute = output0;
    {
      float compute_local[8];

#pragma unroll
      for (int xx_c_init = 0; xx_c_init < 2; ++xx_c_init) {
        compute_local[xx_c_init] = 0.000000e+00f;
        compute_local[(xx_c_init + 2)] = 0.000000e+00f;
        compute_local[(xx_c_init + 4)] = 0.000000e+00f;
        compute_local[(xx_c_init + 6)] = 0.000000e+00f;
      }
#pragma unroll
      for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
        __syncthreads();
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 8;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
          pad_temp_shared[(
              ((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 8)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] =
              input0[(
                  ((((rc_outer * 16384) + (((int)threadIdx.z) * 4096)) +
                    ((((((int)threadIdx.x) * 8) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >>
                      5) *
                     1024)) +
                   (((int)blockIdx.y) * 32)) +
                  (((((int)threadIdx.x) * 8) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) &
                   31))];
        }
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 4;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
          input1_shared[(
              ((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 4)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] =
              input1[(
                  ((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 128)) +
                    ((((((int)threadIdx.x) * 4) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >>
                      4) *
                     32)) +
                   (rc_outer * 16)) +
                  (((((int)threadIdx.x) * 4) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) &
                   15))];
        }
        __syncthreads();
#pragma unroll
        for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
#pragma unroll
          for (int xx_c = 0; xx_c < 2; ++xx_c) {
            compute_local[xx_c] =
                (compute_local[xx_c] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.x) * 2)) + xx_c)] *
                  input1_shared[((((int)threadIdx.z) * 16) + rc_inner)]));
            compute_local[(xx_c + 2)] =
                (compute_local[(xx_c + 2)] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.x) * 2)) + xx_c)] *
                  input1_shared[(((((int)threadIdx.z) * 16) + rc_inner) +
                                 64)]));
            compute_local[(xx_c + 4)] =
                (compute_local[(xx_c + 4)] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.x) * 2)) + xx_c)] *
                  input1_shared[(((((int)threadIdx.z) * 16) + rc_inner) +
                                 128)]));
            compute_local[(xx_c + 6)] =
                (compute_local[(xx_c + 6)] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.x) * 2)) + xx_c)] *
                  input1_shared[(((((int)threadIdx.z) * 16) + rc_inner) +
                                 192)]));
          }
        }
      }
#pragma unroll
      for (int xx_inner_inner_inner = 0; xx_inner_inner_inner < 2;
           ++xx_inner_inner_inner) {
        compute[(((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) +
                   (((int)blockIdx.y) * 32)) +
                  (((int)threadIdx.x) * 2)) +
                 xx_inner_inner_inner)] = compute_local[xx_inner_inner_inner];
        compute[(
            (((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) +
               (((int)blockIdx.y) * 32)) +
              (((int)threadIdx.x) * 2)) +
             xx_inner_inner_inner) +
            4096)] = compute_local[(xx_inner_inner_inner + 2)];
        compute[(
            (((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) +
               (((int)blockIdx.y) * 32)) +
              (((int)threadIdx.x) * 2)) +
             xx_inner_inner_inner) +
            8192)] = compute_local[(xx_inner_inner_inner + 4)];
        compute[(
            (((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) +
               (((int)blockIdx.y) * 32)) +
              (((int)threadIdx.x) * 2)) +
             xx_inner_inner_inner) +
            12288)] = compute_local[(xx_inner_inner_inner + 6)];
      }
    }
  }
}
// Node name:	Relu_510
// Description:	Relu
// Input:
//	- name: Slice_500_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_510_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void
Relu_float_float_cuda_Relu_510_block_kernel(float *input0, float *output0,
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
// Node name:	Matched_Pattern_2989
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_507_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Constant_2047_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_2990_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_541_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989_block_kernel(
    float *input0, float *input1, float *input2, float *output0, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 64) {
    return;
  }
  const dim3 blockDim(16, 1, 4);
  const dim3 gridDim(1, 32, 2);
  const dim3 threadIdx(thread_id % 16, 0, thread_id / 16);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 32, block_id / 32);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 2048);
  {
    float *compute = output0;
    {
      float compute1[8];

#pragma unroll
      for (int xx_init = 0; xx_init < 2; ++xx_init) {
        compute1[xx_init] = 0.000000e+00f;
        compute1[(xx_init + 2)] = 0.000000e+00f;
        compute1[(xx_init + 4)] = 0.000000e+00f;
        compute1[(xx_init + 6)] = 0.000000e+00f;
      }
#pragma unroll
      for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
        __syncthreads();
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 8;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
          pad_temp_shared[(
              ((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 8)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] =
              input0[(
                  ((((rc_outer * 16384) + (((int)threadIdx.z) * 4096)) +
                    ((((((int)threadIdx.x) * 8) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >>
                      5) *
                     1024)) +
                   (((int)blockIdx.y) * 32)) +
                  (((((int)threadIdx.x) * 8) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) &
                   31))];
        }
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 4;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
          input1_shared[(
              ((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 4)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] =
              input1[(
                  ((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 128)) +
                    ((((((int)threadIdx.x) * 4) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >>
                      4) *
                     32)) +
                   (rc_outer * 16)) +
                  (((((int)threadIdx.x) * 4) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) &
                   15))];
        }
        __syncthreads();
#pragma unroll
        for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
#pragma unroll
          for (int xx = 0; xx < 2; ++xx) {
            compute1[xx] =
                (compute1[xx] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.x) * 2)) + xx)] *
                  input1_shared[((((int)threadIdx.z) * 16) + rc_inner)]));
            compute1[(xx + 2)] =
                (compute1[(xx + 2)] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.x) * 2)) + xx)] *
                  input1_shared[(((((int)threadIdx.z) * 16) + rc_inner) +
                                 64)]));
            compute1[(xx + 4)] =
                (compute1[(xx + 4)] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.x) * 2)) + xx)] *
                  input1_shared[(((((int)threadIdx.z) * 16) + rc_inner) +
                                 128)]));
            compute1[(xx + 6)] =
                (compute1[(xx + 6)] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.x) * 2)) + xx)] *
                  input1_shared[(((((int)threadIdx.z) * 16) + rc_inner) +
                                 192)]));
          }
        }
      }
#pragma unroll
      for (int i3_inner_inner_inner = 0; i3_inner_inner_inner < 2;
           ++i3_inner_inner_inner) {
        compute[(((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) +
                   (((int)blockIdx.y) * 32)) +
                  (((int)threadIdx.x) * 2)) +
                 i3_inner_inner_inner)] =
            max((compute1[i3_inner_inner_inner] +
                 input2[((((int)blockIdx.z) * 16) + ((int)threadIdx.z))]),
                0.000000e+00f);
        compute[(
            (((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) +
               (((int)blockIdx.y) * 32)) +
              (((int)threadIdx.x) * 2)) +
             i3_inner_inner_inner) +
            4096)] =
            max((compute1[(i3_inner_inner_inner + 2)] +
                 input2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 4)]),
                0.000000e+00f);
        compute[(
            (((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) +
               (((int)blockIdx.y) * 32)) +
              (((int)threadIdx.x) * 2)) +
             i3_inner_inner_inner) +
            8192)] =
            max((compute1[(i3_inner_inner_inner + 4)] +
                 input2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8)]),
                0.000000e+00f);
        compute[(
            (((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) +
               (((int)blockIdx.y) * 32)) +
              (((int)threadIdx.x) * 2)) +
             i3_inner_inner_inner) +
            12288)] =
            max((compute1[(i3_inner_inner_inner + 6)] +
                 input2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) +
                         12)]),
                0.000000e+00f);
      }
    }
  }
}
// Node name:	Add_508
// Description:	Add
// Input:
//	- name: AvgPool_499_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_499_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_508_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void Add_float_float_float_cuda_Add_508_block_kernel(
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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Relu_Matched_Pattern_Matched_Pattern_Add_Matched_Pattern_Convolution_3(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *input12, float *input13,
    float *input14, float *input15, float *input16, float *output0,
    float *output1, float *output2, float *output3, float *output4,
    float *output5, float *output6, float *output7) {

  __shared__ char shared_buffer[3072];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Relu_float_float_cuda_Relu_510_block_kernel(input4, output2, threadIdx.x,
                                                blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Add_float_float_float_cuda_Add_508_block_kernel(
        input11, input11, output5, threadIdx.x, blockIdx.x - 64, shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    Convolution_float_float_float_cuda_Convolution_537_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 128, shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    Convolution_float_float_float_cuda_Convolution_537_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 192, shared_buffer);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989_block_kernel(
        input5, input6, input7, output3, threadIdx.x, blockIdx.x - 256,
        shared_buffer);
  } else if ((int)blockIdx.x >= 320 && (int)blockIdx.x <= 383) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989_block_kernel(
        input8, input9, input10, output4, threadIdx.x, blockIdx.x - 320,
        shared_buffer);
  } else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 447) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989_block_kernel(
        input12, input13, input14, output6, threadIdx.x, blockIdx.x - 384,
        shared_buffer);
  } else if ((int)blockIdx.x >= 448 && (int)blockIdx.x <= 511) {
    Convolution_float_float_float_cuda_Convolution_537_block_kernel(
        input15, input16, output7, threadIdx.x, blockIdx.x - 448,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Relu_Matched_Pattern_Matched_Pattern_Add_Matched_Pattern_Convolution_3_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *input12, float *input13,
    float *input14, float *input15, float *input16, float *output0,
    float *output1, float *output2, float *output3, float *output4,
    float *output5, float *output6, float *output7) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Relu_Matched_Pattern_Matched_Pattern_Add_Matched_Pattern_Convolution_3<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, input11, input12, input13, input14, input15, input16,
      output0, output1, output2, output3, output4, output5, output6, output7);
}
// Node name:	Add_2724
// Description:	Add
// Input:
//	- name: Convolution_1729_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2852_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1730_0	type: float	shape: Shape{1,
// 128, 8, 8}
extern "C" __launch_bounds__(512) __global__
    void Add_float_float_float_cuda_Add_2724(float *input0, float *input1,
                                             float *output0) {
  output0[blockIdx.x * 512 + threadIdx.x] =
      add(input0[blockIdx.x * 512 + threadIdx.x],
          input1[blockIdx.x * 512 + threadIdx.x]);
}
extern void Add_float_float_float_cuda_Add_2724_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *output0) {
  Add_float_float_float_cuda_Add_2724<<<grids, blocks, mem, stream>>>(
      input0, input1, output0);
}
// Node name:	MaxPool_1351
// Description:	MaxPool
// Input:
//	- name: Slice_1347_0	type: float	shape: Shape{1, 128, 16, 16}
// Output:
//	- name: MaxPool_1351_0	type: float	shape: Shape{1, 128, 8, 8}
void MaxPool_float_float_cuda_lib_MaxPool_1351(cudnnHandle_t cudnn_handle,
                                               float *input0, float *output0) {
  cudnnTensorDescriptor_t input_desc;
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, 1, 128, 16, 16));
  cudnnTensorDescriptor_t output_desc;
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
  CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, 1, 128, 8, 8));
  cudnnPoolingDescriptor_t desc;
  cudnnCreatePoolingDescriptor(&desc);
  CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(
      desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 3, 3, 0, 0, 2, 2));
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
//	- name: Relu_1444_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_331_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Relu_1446_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_124_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Relu_1445_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_295_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Relu_1426_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_112_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Constant_104_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: AvgPool_1427_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1357_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Output:
//	- name: DepthwiseConv2dNative_1449_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1451_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1450_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1434_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1435_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Add_1436_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1449<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1444_0, Constant_331_0,
// DepthwiseConv2dNative_1449_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1451<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1446_0, Constant_124_0,
// DepthwiseConv2dNative_1451_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1450<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1445_0, Constant_295_0,
// DepthwiseConv2dNative_1450_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1434<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1426_0, Constant_112_0,
// DepthwiseConv2dNative_1434_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1435<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1426_0, Constant_104_0,
// DepthwiseConv2dNative_1435_0); Add_float_float_float_cuda_Add_1436<<<dim3(16,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1427_0, BatchNormInference_1357_0,
// Add_1436_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1450 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1449
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1434 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1449
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1435 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1451

// Node name:	DepthwiseConv2dNative_1449
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1444_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_331_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1449_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1449_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1451
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1446_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_124_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1451_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1451_block_kernel(
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
// Node name:	Add_1436
// Description:	Add
// Input:
//	- name: AvgPool_1427_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1357_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Output:
//	- name: Add_1436_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_1436_block_kernel(float *input0, float *input1,
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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_134(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    Add_float_float_float_cuda_Add_1436_block_kernel(
        input9, input10, output5, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 79) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1449_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 16, NULL);
  } else if ((int)blockIdx.x >= 80 && (int)blockIdx.x <= 143) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1451_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 80, NULL);
  } else if ((int)blockIdx.x >= 144 && (int)blockIdx.x <= 207) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1449_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 144, NULL);
  } else if ((int)blockIdx.x >= 208 && (int)blockIdx.x <= 271) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1449_block_kernel(
        input6, input7, output3, threadIdx.x, blockIdx.x - 208, NULL);
  } else if ((int)blockIdx.x >= 272 && (int)blockIdx.x <= 335) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1451_block_kernel(
        input6, input8, output4, threadIdx.x, blockIdx.x - 272, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_134_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_134<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4, output5);
}
