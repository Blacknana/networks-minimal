// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_2828
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2828_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2828(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2828_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2828_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_156
// Description:	Constant
// Input:
// Output:
//	- name: Constant_156_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_156(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_156_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_156_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_295
// Description:	Constant
// Input:
// Output:
//	- name: Constant_295_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_295(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_295_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_295_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2173
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2173_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2173(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2173_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2173_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2796
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2796_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2796(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2796_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2796_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2755
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2755_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2755(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2755_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2755_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_321
// Description:	Constant
// Input:
// Output:
//	- name: Constant_321_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_321(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_321_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_321_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_100
// Description:	Constant
// Input:
// Output:
//	- name: Constant_100_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_100(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_100_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_100_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2395
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2395_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2395(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2395_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2395_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2557
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2557_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2557(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2557_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2557_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1158_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_267_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Constant_340_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Constant_133_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: BatchNormInference_1156_0	type: float	shape: Shape{1,
//64, 16, 16}
// Output:
//	- name: DepthwiseConv2dNative_1161_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: DepthwiseConv2dNative_1162_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: DepthwiseConv2dNative_1163_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Slice_1159_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1161<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1158_0, Constant_267_0,
// DepthwiseConv2dNative_1161_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1162<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1158_0, Constant_340_0,
// DepthwiseConv2dNative_1162_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1163<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1158_0, Constant_133_0,
// DepthwiseConv2dNative_1163_0); Slice_float_float_cuda_Slice_1159<<<dim3(256,
// 1, 1), dim3(64, 1, 1), 0, 0>>>(BatchNormInference_1156_0, Slice_1159_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1163 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1161

// Node name:	DepthwiseConv2dNative_1161
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1158_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_267_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1161_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ static void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1161_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1162
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1158_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_340_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1162_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ static void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1162_block_kernel(
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
// Node name:	Slice_1159
// Description:	Slice
// Input:
//	- name: BatchNormInference_1156_0	type: float	shape: Shape{1,
//64, 16, 16}
// Output:
//	- name: Slice_1159_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ static void
Slice_float_float_cuda_Slice_1159_block_kernel(float *input0, float *output0,
                                               int thread_id, int block_id,
                                               char *shared_buffer) {
  if (thread_id >= 64) {
    return;
  }
  const dim3 blockDim(64, 1, 1);
  const dim3 gridDim(256, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < 16384) {
    uint32_t input_strides[] = {16384, 256, 16, 1};
    uint32_t output_strides[] = {16384, 256, 16, 1};
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_94(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1161_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 255) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1162_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 128 + 0, NULL);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 383) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1161_block_kernel(
        input0, input3, output2, threadIdx.x, blockIdx.x - 256 + 0, NULL);
  } else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 639) {
    Slice_float_float_cuda_Slice_1159_block_kernel(input4, output3, threadIdx.x,
                                                   blockIdx.x - 384 + 0, NULL);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_94_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_94<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    output0, output1, output2, output3);
}
// Node name:	Constant_144
// Description:	Constant
// Input:
// Output:
//	- name: Constant_144_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_144(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_144_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_144_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1276_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2473_0	type: float	shape: Shape{128, 384, 1, 1}
//	- name: Constant_3108_0	type: float	shape: Shape{1, 128, 16, 16}
//	- name: Constant_2476_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Relu_1283_0	type: float	shape: Shape{1, 128, 16, 16}
//	- name: Convolution_1280_0	type: float	shape: Shape{1, 64, 16,
//16}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3107<<<dim3(2,
// 8, 4), dim3(8, 2, 16), 0, 0>>>(Relu_1276_0, Constant_2473_0, Constant_3108_0,
// Relu_1283_0); Convolution_float_float_float_cuda_Convolution_1280<<<dim3(1,
// 16, 4), dim3(16, 1, 16), 0, 0>>>(Relu_1276_0, Constant_2476_0,
// Convolution_1280_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	Matched_Pattern_3107
// Description:	Matched_Pattern
// Input:
//	- name: Relu_1276_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2473_0	type: float	shape: Shape{128, 384, 1, 1}
//	- name: Constant_3108_0	type: float	shape: Shape{1, 128, 16, 16}
// Output:
//	- name: Relu_1283_0	type: float	shape: Shape{1, 128, 16, 16}
__device__ __forceinline__ static void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3107_block_kernel(
    float *input0, float *input1, float *input2, float *output0, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(8, 2, 16);
  const dim3 gridDim(2, 8, 4);
  const dim3 threadIdx(thread_id % 8, thread_id / 8 % 2, thread_id / 16);
  const dim3 blockIdx(block_id % 2, block_id / 2 % 8, block_id / 16);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 2048);
  {
    float *compute = output0;
    {
      float compute1[2];

      compute1[0] = 0.000000e+00f;
      compute1[1] = 0.000000e+00f;
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                     (((int)blockIdx.y) * 32)) +
                    ((((int)threadIdx.x) >> 2) * 16)) +
                   (((int)blockIdx.x) * 8)) +
                  ((((int)threadIdx.x) & 3) * 2))];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                     (((int)blockIdx.y) * 32)) +
                    ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) +
                   (((int)blockIdx.x) * 8)) +
                  (((((int)threadIdx.x) * 2) + 1) & 7))];
      input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                     (((int)threadIdx.x) * 4))] =
          input1[((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                   (((int)threadIdx.y) * 384)) +
                  (((int)threadIdx.x) * 4))];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  1)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  2)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     3)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  3)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[((((int)threadIdx.z) * 32) + 512)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 513)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 514)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 515)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 516)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 517)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 518)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 519)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 520)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 521)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 522)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 523)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 524)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 525)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 526)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 527)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 528)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 529)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 530)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 531)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 532)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 533)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 534)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 535)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 536)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 537)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 538)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 539)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 540)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 541)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 542)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 543)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((int)threadIdx.x) >> 2) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((((int)threadIdx.x) & 3) * 2)) +
              8192)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) +
                (((int)blockIdx.x) * 8)) +
               (((((int)threadIdx.x) * 2) + 1) & 7)) +
              8192)];
      input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                     (((int)threadIdx.x) * 4))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  32)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  33)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  34)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     3)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  35)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[((((int)threadIdx.z) * 32) + 512)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 513)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 514)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 515)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 516)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 517)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 518)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 519)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 520)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 521)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 522)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 523)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 524)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 525)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 526)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 527)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 528)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 529)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 530)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 531)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 532)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 533)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 534)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 535)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 536)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 537)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 538)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 539)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 540)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 541)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 542)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 543)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((int)threadIdx.x) >> 2) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((((int)threadIdx.x) & 3) * 2)) +
              16384)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) +
                (((int)blockIdx.x) * 8)) +
               (((((int)threadIdx.x) * 2) + 1) & 7)) +
              16384)];
      input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                     (((int)threadIdx.x) * 4))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  64)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  65)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  66)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     3)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  67)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[((((int)threadIdx.z) * 32) + 512)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 513)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 514)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 515)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 516)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 517)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 518)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 519)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 520)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 521)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 522)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 523)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 524)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 525)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 526)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 527)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 528)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 529)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 530)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 531)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 532)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 533)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 534)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 535)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 536)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 537)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 538)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 539)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 540)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 541)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 542)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 543)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((int)threadIdx.x) >> 2) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((((int)threadIdx.x) & 3) * 2)) +
              24576)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) +
                (((int)blockIdx.x) * 8)) +
               (((((int)threadIdx.x) * 2) + 1) & 7)) +
              24576)];
      input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                     (((int)threadIdx.x) * 4))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  96)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  97)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  98)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     3)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  99)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[((((int)threadIdx.z) * 32) + 512)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 513)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 514)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 515)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 516)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 517)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 518)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 519)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 520)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 521)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 522)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 523)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 524)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 525)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 526)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 527)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 528)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 529)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 530)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 531)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 532)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 533)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 534)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 535)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 536)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 537)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 538)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 539)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 540)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 541)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 542)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 543)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((int)threadIdx.x) >> 2) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((((int)threadIdx.x) & 3) * 2)) +
              32768)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) +
                (((int)blockIdx.x) * 8)) +
               (((((int)threadIdx.x) * 2) + 1) & 7)) +
              32768)];
      input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                     (((int)threadIdx.x) * 4))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  128)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  129)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  130)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     3)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  131)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[((((int)threadIdx.z) * 32) + 512)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 513)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 514)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 515)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 516)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 517)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 518)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 519)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 520)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 521)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 522)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 523)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 524)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 525)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 526)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 527)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 528)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 529)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 530)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 531)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 532)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 533)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 534)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 535)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 536)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 537)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 538)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 539)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 540)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 541)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 542)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 543)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((int)threadIdx.x) >> 2) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((((int)threadIdx.x) & 3) * 2)) +
              40960)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) +
                (((int)blockIdx.x) * 8)) +
               (((((int)threadIdx.x) * 2) + 1) & 7)) +
              40960)];
      input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                     (((int)threadIdx.x) * 4))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  160)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  161)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  162)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     3)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  163)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[((((int)threadIdx.z) * 32) + 512)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 513)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 514)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 515)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 516)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 517)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 518)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 519)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 520)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 521)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 522)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 523)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 524)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 525)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 526)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 527)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 528)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 529)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 530)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 531)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 532)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 533)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 534)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 535)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 536)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 537)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 538)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 539)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 540)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 541)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 542)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 543)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((int)threadIdx.x) >> 2) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((((int)threadIdx.x) & 3) * 2)) +
              49152)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) +
                (((int)blockIdx.x) * 8)) +
               (((((int)threadIdx.x) * 2) + 1) & 7)) +
              49152)];
      input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                     (((int)threadIdx.x) * 4))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  192)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  193)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  194)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     3)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  195)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[((((int)threadIdx.z) * 32) + 512)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 513)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 514)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 515)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 516)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 517)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 518)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 519)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 520)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 521)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 522)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 523)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 524)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 525)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 526)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 527)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 528)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 529)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 530)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 531)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 532)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 533)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 534)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 535)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 536)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 537)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 538)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 539)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 540)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 541)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 542)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 543)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((int)threadIdx.x) >> 2) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((((int)threadIdx.x) & 3) * 2)) +
              57344)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) +
                (((int)blockIdx.x) * 8)) +
               (((((int)threadIdx.x) * 2) + 1) & 7)) +
              57344)];
      input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                     (((int)threadIdx.x) * 4))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  224)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  225)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  226)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     3)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  227)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[((((int)threadIdx.z) * 32) + 512)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 513)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 514)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 515)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 516)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 517)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 518)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 519)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 520)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 521)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 522)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 523)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 524)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 525)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 526)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 527)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 528)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 529)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 530)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 531)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 532)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 533)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 534)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 535)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 536)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 537)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 538)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 539)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 540)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 541)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 542)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 543)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((int)threadIdx.x) >> 2) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((((int)threadIdx.x) & 3) * 2)) +
              65536)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) +
                (((int)blockIdx.x) * 8)) +
               (((((int)threadIdx.x) * 2) + 1) & 7)) +
              65536)];
      input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                     (((int)threadIdx.x) * 4))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  256)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  257)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  258)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     3)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  259)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[((((int)threadIdx.z) * 32) + 512)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 513)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 514)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 515)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 516)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 517)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 518)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 519)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 520)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 521)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 522)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 523)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 524)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 525)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 526)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 527)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 528)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 529)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 530)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 531)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 532)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 533)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 534)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 535)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 536)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 537)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 538)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 539)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 540)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 541)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 542)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 543)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((int)threadIdx.x) >> 2) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((((int)threadIdx.x) & 3) * 2)) +
              73728)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) +
                (((int)blockIdx.x) * 8)) +
               (((((int)threadIdx.x) * 2) + 1) & 7)) +
              73728)];
      input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                     (((int)threadIdx.x) * 4))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  288)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  289)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  290)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     3)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  291)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[((((int)threadIdx.z) * 32) + 512)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 513)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 514)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 515)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 516)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 517)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 518)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 519)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 520)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 521)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 522)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 523)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 524)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 525)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 526)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 527)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 528)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 529)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 530)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 531)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 532)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 533)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 534)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 535)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 536)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 537)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 538)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 539)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 540)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 541)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 542)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 543)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((int)threadIdx.x) >> 2) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((((int)threadIdx.x) & 3) * 2)) +
              81920)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) +
                (((int)blockIdx.x) * 8)) +
               (((((int)threadIdx.x) * 2) + 1) & 7)) +
              81920)];
      input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                     (((int)threadIdx.x) * 4))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  320)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  321)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  322)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     3)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  323)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[((((int)threadIdx.z) * 32) + 512)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 513)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 514)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 515)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 516)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 517)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 518)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 519)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 520)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 521)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 522)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 523)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 524)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 525)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 526)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 527)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 528)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 529)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 530)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 531)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 532)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 533)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 534)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 535)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 536)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 537)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 538)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 539)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 540)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 541)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 542)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 543)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((int)threadIdx.x) >> 2) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((((int)threadIdx.x) & 3) * 2)) +
              90112)];
      pad_temp_shared[(
          (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
           (((int)threadIdx.x) * 2)) +
          1)] =
          input0[(
              ((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) +
                (((int)blockIdx.x) * 8)) +
               (((((int)threadIdx.x) * 2) + 1) & 7)) +
              90112)];
      input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                     (((int)threadIdx.x) * 4))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  352)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  353)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  354)];
      input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) +
                      (((int)threadIdx.x) * 4)) +
                     3)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 384)) +
                   (((int)threadIdx.x) * 4)) +
                  355)];
      __syncthreads();
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] *
            input1_shared[((((int)threadIdx.z) * 32) + 512)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 513)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 514)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 515)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 516)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 517)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 518)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 519)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 520)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 521)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 522)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 523)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 524)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 525)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 526)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 527)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 528)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 529)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 530)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 531)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 532)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 533)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 534)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 535)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 536)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 537)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 538)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 539)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 540)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 541)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 542)]));
      compute1[0] =
          (compute1[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute1[1] =
          (compute1[1] +
           (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 543)]));
      compute[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 256)) +
                  (((int)blockIdx.y) * 32)) +
                 (((int)threadIdx.y) * 16)) +
                (((int)blockIdx.x) * 8)) +
               ((int)threadIdx.x))] =
          max((compute1[0] +
               input2[((((int)blockIdx.z) * 32) + ((int)threadIdx.z))]),
              0.000000e+00f);
      compute[(((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 256)) +
                   (((int)blockIdx.y) * 32)) +
                  (((int)threadIdx.y) * 16)) +
                 (((int)blockIdx.x) * 8)) +
                ((int)threadIdx.x)) +
               4096)] =
          max((compute1[1] +
               input2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16)]),
              0.000000e+00f);
    }
  }
}
// Node name:	Convolution_1280
// Description:	Convolution
// Input:
//	- name: Relu_1276_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2476_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1280_0	type: float	shape: Shape{1, 64, 16,
//16}
__device__ __forceinline__ static void
Convolution_float_float_float_cuda_Convolution_1280_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_cuda_Matched_Pattern_Convolution_110(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  __shared__ char shared_buffer[6144];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3107_block_kernel(
        input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_1280_block_kernel(
        input0, input3, output1, threadIdx.x, blockIdx.x - 64 + 0,
        shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_cuda_Matched_Pattern_Convolution_110_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_cuda_Matched_Pattern_Convolution_110<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2925_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1328_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: BatchNormInference_1269_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2926_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1330_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Slice_1284_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1333_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_1334_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_48<<<dim3(32, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1328_0, Constant_2925_0,
// BatchNormInference_1269_0, Add_1333_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_49<<<dim3(32, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1330_0, Constant_2926_0, Slice_1284_0,
// Add_1334_0); Deduped function map: <src_function_name :
// deduped_function_name> FusedKernel_float_float_float_float_cuda_Add_Add_49 :
// FusedKernel_float_float_float_float_cuda_Add_Add_48

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1328_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2925_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1269_0	type: float	shape: Shape{1,
//64, 16, 16}
// Output:
//	- name: Add_1333_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2505<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1328_0, Constant_2925_0, BatchNormInference_1331_0);
// Add_float_float_float_cuda_Add_1333<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1331_0, BatchNormInference_1269_0, Add_1333_0);
__device__ __forceinline__ static void
FusedKernel_float_float_float_float_cuda_Add_Add_48_block_kernel(
    float *input0, float *input1, float *input2, float *output0, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(32, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = add(temp0, input2[tid]);
  output0[tid] = temp1;
}
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_118(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    FusedKernel_float_float_float_float_cuda_Add_Add_48_block_kernel(
        input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    FusedKernel_float_float_float_float_cuda_Add_Add_48_block_kernel(
        input4, input3, input5, output1, threadIdx.x, blockIdx.x - 32 + 0,
        NULL);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_118_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_118<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, output0, output1);
}
// Node name:	Result_1763
// Description:	Result
// Input:
//	- name: Add_1762_0	type: float	shape: Shape{1, 10}
// Output:
//	- name: Result_1763_0	type: float	shape: Shape{1, 10}
void Result_float_float_cuda_lib_Result_1763(float *input0, float **output0) {
  *output0 = input0;
}
