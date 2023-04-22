// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	Constant_2062
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2062_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2062(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2062_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2062_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2377
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2377_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2377(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2377_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2377_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3056
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3056_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3056(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3056_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3056_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2917
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2917_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2917(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2917_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2917_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2946
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2946_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2946(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2946_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2946_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_387
// Description:	Constant
// Input:
// Output:
//	- name: Constant_387_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_387(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_387_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_387_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_323
// Description:	Constant
// Input:
// Output:
//	- name: Constant_323_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_323(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_323_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_323_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_436
// Description:	Constant
// Input:
// Output:
//	- name: Constant_436_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_436(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_436_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_436_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2269
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2269_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2269(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2269_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2269_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2879
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2879_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2879(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2879_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2879_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1031_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_212_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Constant_225_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Constant_265_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: BatchNormInference_1030_0	type: float	shape: Shape{1,
// 64, 16, 16}
// Output:
//	- name: DepthwiseConv2dNative_1036_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: DepthwiseConv2dNative_1034_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: DepthwiseConv2dNative_1035_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Slice_1033_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1036<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1031_0, Constant_212_0,
// DepthwiseConv2dNative_1036_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1034<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1031_0, Constant_225_0,
// DepthwiseConv2dNative_1034_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1035<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1031_0, Constant_265_0,
// DepthwiseConv2dNative_1035_0); Slice_float_float_cuda_Slice_1033<<<dim3(256,
// 1, 1), dim3(64, 1, 1), 0, 0>>>(BatchNormInference_1030_0, Slice_1033_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1035 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1036

// Node name:	DepthwiseConv2dNative_1036
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1031_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_212_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1036_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1036_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1034
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1031_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_225_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1034_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1034_block_kernel(
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
// Node name:	Slice_1033
// Description:	Slice
// Input:
//	- name: BatchNormInference_1030_0	type: float	shape: Shape{1,
// 64, 16, 16}
// Output:
//	- name: Slice_1033_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Slice_float_float_cuda_Slice_1033_block_kernel(float *input0, float *output0,
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_76(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1036_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 255) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1034_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 128, NULL);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 383) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1036_block_kernel(
        input0, input3, output2, threadIdx.x, blockIdx.x - 256, NULL);
  } else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 639) {
    Slice_float_float_cuda_Slice_1033_block_kernel(input4, output3, threadIdx.x,
                                                   blockIdx.x - 384, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_76_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_76<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_825_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2224_0	type: float	shape: Shape{64, 192, 1, 1}
//	- name: Constant_3036_0	type: float	shape: Shape{1, 64, 32, 32}
//	- name: Constant_2227_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Relu_832_0	type: float	shape: Shape{1, 64, 32, 32}
//	- name: Convolution_829_0	type: float	shape: Shape{1, 32, 32,
// 32}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3035<<<dim3(1,
// 32, 2), dim3(16, 1, 16), 0, 0>>>(Relu_825_0, Constant_2224_0,
// Constant_3036_0, Relu_832_0);
// Convolution_float_float_float_cuda_Convolution_829<<<dim3(1, 32, 2), dim3(32,
// 1, 8), 0, 0>>>(Relu_825_0, Constant_2227_0, Convolution_829_0); Deduped
// function map: <src_function_name : deduped_function_name>

// Node name:	Matched_Pattern_3035
// Description:	Matched_Pattern
// Input:
//	- name: Relu_825_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2224_0	type: float	shape: Shape{64, 192, 1, 1}
//	- name: Constant_3036_0	type: float	shape: Shape{1, 64, 32, 32}
// Output:
//	- name: Relu_832_0	type: float	shape: Shape{1, 64, 32, 32}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3035_block_kernel(
    float *input0, float *input1, float *input2, float *output0, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(16, 1, 16);
  const dim3 gridDim(1, 32, 2);
  const dim3 threadIdx(thread_id % 16, 0, thread_id / 16);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 32, block_id / 32);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 3072);
  {
    float *compute = output0;
    {
      float compute1[4];

#pragma unroll
      for (int ff_init = 0; ff_init < 2; ++ff_init) {
        compute1[ff_init] = 0.000000e+00f;
        compute1[(ff_init + 2)] = 0.000000e+00f;
      }
#pragma unroll
      for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
        __syncthreads();
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 3;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
          pad_temp_shared[(
              ((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] =
              relu(input0[(
                  (((rc_outer * 24576) +
                    (((((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >>
                      5) *
                     1024)) +
                   (((int)blockIdx.y) * 32)) +
                  ((((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) &
                   31))]);
        }
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
          input1_shared[(
              ((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] =
              input1[(
                  ((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) /
                      24) *
                     192)) +
                   (rc_outer * 24)) +
                  (((((int)threadIdx.x) * 3) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) %
                   24))];
        }
        __syncthreads();
#pragma unroll
        for (int rc_inner = 0; rc_inner < 24; ++rc_inner) {
#pragma unroll
          for (int ff = 0; ff < 2; ++ff) {
            compute1[ff] =
                (compute1[ff] +
                 (pad_temp_shared[((rc_inner * 32) + ((int)threadIdx.x))] *
                  input1_shared[(((((int)threadIdx.z) * 48) + (ff * 24)) +
                                 rc_inner)]));
            compute1[(ff + 2)] =
                (compute1[(ff + 2)] +
                 (pad_temp_shared[(((rc_inner * 32) + ((int)threadIdx.x)) +
                                   16)] *
                  input1_shared[(((((int)threadIdx.z) * 48) + (ff * 24)) +
                                 rc_inner)]));
          }
        }
      }
#pragma unroll
      for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 2;
           ++i1_inner_inner_inner) {
        compute[(((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) +
                   (i1_inner_inner_inner * 1024)) +
                  (((int)blockIdx.y) * 32)) +
                 ((int)threadIdx.x))] =
            max((compute1[i1_inner_inner_inner] +
                 input2[(((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) +
                         i1_inner_inner_inner)]),
                0.000000e+00f);
        compute[(
            (((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) +
               (i1_inner_inner_inner * 1024)) +
              (((int)blockIdx.y) * 32)) +
             ((int)threadIdx.x)) +
            16)] =
            max((compute1[(i1_inner_inner_inner + 2)] +
                 input2[(((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) +
                         i1_inner_inner_inner)]),
                0.000000e+00f);
      }
    }
  }
}
// Node name:	Convolution_829
// Description:	Convolution
// Input:
//	- name: Relu_825_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2227_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_829_0	type: float	shape: Shape{1, 32, 32,
// 32}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_829_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(32, 1, 8);
  const dim3 gridDim(1, 32, 2);
  const dim3 threadIdx(thread_id % 32, 0, thread_id / 32);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 32, block_id / 32);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 6144);
  {
    float *compute = output0;
    {
      float compute_local[2];

      compute_local[0] = 0.000000e+00f;
      compute_local[1] = 0.000000e+00f;
      pad_temp_shared[((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6))] =
          relu(input0[((((((int)threadIdx.z) * 6144) +
                         (((((int)threadIdx.x) * 6) / 32) * 1024)) +
                        (((int)blockIdx.y) * 32)) +
                       ((((int)threadIdx.x) * 6) & 31))]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       1)] =
          relu(input0[((((((int)threadIdx.z) * 6144) +
                         ((((((int)threadIdx.x) * 6) + 1) >> 5) * 1024)) +
                        (((int)blockIdx.y) * 32)) +
                       (((((int)threadIdx.x) * 6) + 1) & 31))]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       2)] =
          relu(input0[((((((int)threadIdx.z) * 6144) +
                         ((((((int)threadIdx.x) * 6) + 2) >> 5) * 1024)) +
                        (((int)blockIdx.y) * 32)) +
                       (((((int)threadIdx.x) * 6) + 2) & 31))]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       3)] =
          relu(input0[((((((int)threadIdx.z) * 6144) +
                         ((((((int)threadIdx.x) * 6) + 3) >> 5) * 1024)) +
                        (((int)blockIdx.y) * 32)) +
                       (((((int)threadIdx.x) * 6) + 3) & 31))]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       4)] =
          relu(input0[((((((int)threadIdx.z) * 6144) +
                         ((((((int)threadIdx.x) * 6) + 4) >> 5) * 1024)) +
                        (((int)blockIdx.y) * 32)) +
                       (((((int)threadIdx.x) * 6) + 4) & 31))]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       5)] =
          relu(input0[((((((int)threadIdx.z) * 6144) +
                         ((((((int)threadIdx.x) * 6) + 5) >> 5) * 1024)) +
                        (((int)blockIdx.y) * 32)) +
                       (((((int)threadIdx.x) * 6) + 5) & 31))]);
      input1_shared[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] =
          input1[((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                   ((((int)threadIdx.x) >> 4) * 192)) +
                  ((((int)threadIdx.x) & 15) * 3))];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                   ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) +
                  (((((int)threadIdx.x) * 3) + 1) % 48))];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                   ((((((int)threadIdx.x) * 3) + 2) / 48) * 192)) +
                  (((((int)threadIdx.x) * 3) + 2) % 48))];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 96)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((int)threadIdx.x)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 49)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 50)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 51)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 52)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 53)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 54)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 55)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 56)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 57)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 58)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 59)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 60)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 61)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 62)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 63)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 16)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 64)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 17)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 65)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 18)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 66)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 19)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 67)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 20)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 68)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 21)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 69)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 22)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 70)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 23)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 71)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 24)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 72)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 25)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 73)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 26)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 74)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 27)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 75)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 28)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 76)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 29)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 77)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 30)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 78)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 31)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 79)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 32)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 80)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 33)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 81)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 34)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 82)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 35)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 83)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 36)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 84)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 37)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 85)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 38)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 86)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 39)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 87)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 40)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 88)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 41)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 89)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 42)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 90)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 43)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 91)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 44)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 92)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 45)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 93)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 46)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 94)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 47)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 95)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6))] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          (((((int)threadIdx.x) * 6) / 32) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        ((((int)threadIdx.x) * 6) & 31)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 1) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 1) & 31)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 2) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 2) & 31)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       3)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 3) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 3) & 31)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       4)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 4) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 4) & 31)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       5)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 5) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 5) & 31)) +
                       49152)]);
      input1_shared[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((int)threadIdx.x) >> 4) * 192)) +
                   ((((int)threadIdx.x) & 15) * 3)) +
                  48)];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 1) % 48)) +
                  48)];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 2) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 2) % 48)) +
                  48)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 96)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((int)threadIdx.x)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 49)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 50)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 51)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 52)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 53)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 54)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 55)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 56)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 57)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 58)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 59)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 60)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 61)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 62)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 63)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 16)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 64)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 17)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 65)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 18)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 66)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 19)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 67)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 20)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 68)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 21)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 69)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 22)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 70)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 23)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 71)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 24)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 72)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 25)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 73)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 26)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 74)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 27)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 75)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 28)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 76)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 29)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 77)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 30)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 78)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 31)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 79)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 32)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 80)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 33)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 81)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 34)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 82)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 35)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 83)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 36)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 84)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 37)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 85)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 38)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 86)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 39)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 87)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 40)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 88)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 41)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 89)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 42)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 90)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 43)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 91)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 44)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 92)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 45)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 93)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 46)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 94)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 47)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 95)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6))] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          (((((int)threadIdx.x) * 6) / 32) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        ((((int)threadIdx.x) * 6) & 31)) +
                       98304)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 1) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 1) & 31)) +
                       98304)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 2) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 2) & 31)) +
                       98304)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       3)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 3) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 3) & 31)) +
                       98304)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       4)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 4) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 4) & 31)) +
                       98304)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       5)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 5) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 5) & 31)) +
                       98304)]);
      input1_shared[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((int)threadIdx.x) >> 4) * 192)) +
                   ((((int)threadIdx.x) & 15) * 3)) +
                  96)];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 1) % 48)) +
                  96)];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 2) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 2) % 48)) +
                  96)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 96)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((int)threadIdx.x)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 49)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 50)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 51)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 52)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 53)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 54)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 55)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 56)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 57)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 58)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 59)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 60)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 61)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 62)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 63)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 16)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 64)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 17)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 65)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 18)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 66)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 19)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 67)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 20)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 68)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 21)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 69)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 22)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 70)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 23)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 71)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 24)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 72)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 25)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 73)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 26)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 74)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 27)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 75)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 28)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 76)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 29)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 77)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 30)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 78)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 31)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 79)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 32)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 80)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 33)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 81)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 34)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 82)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 35)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 83)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 36)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 84)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 37)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 85)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 38)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 86)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 39)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 87)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 40)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 88)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 41)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 89)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 42)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 90)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 43)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 91)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 44)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 92)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 45)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 93)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 46)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 94)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 47)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 95)]));
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6))] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          (((((int)threadIdx.x) * 6) / 32) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        ((((int)threadIdx.x) * 6) & 31)) +
                       147456)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 1) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 1) & 31)) +
                       147456)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 2) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 2) & 31)) +
                       147456)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       3)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 3) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 3) & 31)) +
                       147456)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       4)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 4) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 4) & 31)) +
                       147456)]);
      pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) +
                       5)] =
          relu(input0[(((((((int)threadIdx.z) * 6144) +
                          ((((((int)threadIdx.x) * 6) + 5) >> 5) * 1024)) +
                         (((int)blockIdx.y) * 32)) +
                        (((((int)threadIdx.x) * 6) + 5) & 31)) +
                       147456)]);
      input1_shared[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((int)threadIdx.x) >> 4) * 192)) +
                   ((((int)threadIdx.x) & 15) * 3)) +
                  144)];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 1) % 48)) +
                  144)];
      input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                     2)] =
          input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 2) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 2) % 48)) +
                  144)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] *
                               input1_shared[(((int)threadIdx.z) * 96)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((int)threadIdx.x)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 48)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 49)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 50)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 51)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 52)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 53)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 54)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 55)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 56)]));
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                               input1_shared[((((int)threadIdx.z) * 96) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 57)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 58)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 59)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 60)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 61)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 62)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 63)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 16)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 64)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 17)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 65)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 18)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 66)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 19)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 67)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 20)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 68)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 21)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 69)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 22)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 70)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 23)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 71)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 24)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 72)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 25)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 73)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 26)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 74)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 27)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 75)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 28)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 76)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 29)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 77)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 30)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 78)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 31)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 79)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 32)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 80)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 33)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 81)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 34)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 82)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 35)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 83)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 36)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 84)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 37)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 85)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 38)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 86)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 39)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 87)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 40)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 88)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 41)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 89)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 42)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 90)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 43)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 91)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 44)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 92)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 45)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 93)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 46)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 94)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 47)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                           input1_shared[((((int)threadIdx.z) * 96) + 95)]));
      compute[((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) +
                (((int)blockIdx.y) * 32)) +
               ((int)threadIdx.x))] = compute_local[0];
      compute[(((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) +
                 (((int)blockIdx.y) * 32)) +
                ((int)threadIdx.x)) +
               1024)] = compute_local[1];
    }
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_cuda_Matched_Pattern_Convolution_45(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {

  __shared__ char shared_buffer[9216];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3035_block_kernel(
        input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_829_block_kernel(
        input0, input3, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_cuda_Matched_Pattern_Convolution_45_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_cuda_Matched_Pattern_Convolution_45<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Slice_968_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_973_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2299_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3058_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_975_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2305_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3062_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_974_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2302_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3060_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: AvgPool_970_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_971_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_995_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_997_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_996_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_976_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Relu_float_float_cuda_Relu_971<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Slice_968_0, Relu_971_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_973_0, Constant_2299_0,
// Constant_3058_0, Relu_995_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3061<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_975_0, Constant_2305_0,
// Constant_3062_0, Relu_997_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3059<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_974_0, Constant_2302_0,
// Constant_3060_0, Relu_996_0); Add_float_float_float_cuda_Add_976<<<dim3(32,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_970_0, AvgPool_970_0, Add_976_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3061 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3059 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057

// Node name:	Relu_971
// Description:	Relu
// Input:
//	- name: Slice_968_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_971_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Relu_float_float_cuda_Relu_971_block_kernel(float *input0, float *output0,
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
// Node name:	Matched_Pattern_3057
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_973_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2299_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3058_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_995_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057_block_kernel(
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
// Node name:	Add_976
// Description:	Add
// Input:
//	- name: AvgPool_970_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: AvgPool_970_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_976_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void Add_float_float_float_cuda_Add_976_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_68(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {

  __shared__ char shared_buffer[2048];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Relu_float_float_cuda_Relu_971_block_kernel(input0, output0, threadIdx.x,
                                                blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    Add_float_float_float_cuda_Add_976_block_kernel(
        input10, input10, output4, threadIdx.x, blockIdx.x - 32, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057_block_kernel(
        input1, input2, input3, output1, threadIdx.x, blockIdx.x - 64,
        shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057_block_kernel(
        input4, input5, input6, output2, threadIdx.x, blockIdx.x - 128,
        shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057_block_kernel(
        input7, input8, input9, output3, threadIdx.x, blockIdx.x - 192,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_68_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_68<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: AvgPool_1548_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1480_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Relu_1549_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_135_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Constant_338_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Relu_1572_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_455_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Relu_1574_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_39_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Relu_1573_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_196_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: Add_1554_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1555_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1556_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1577_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1579_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1578_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_1554<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(AvgPool_1548_0, BatchNormInference_1480_0, Add_1554_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1555<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1549_0, Constant_135_0,
// DepthwiseConv2dNative_1555_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1556<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1549_0, Constant_338_0,
// DepthwiseConv2dNative_1556_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1577<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1572_0, Constant_455_0,
// DepthwiseConv2dNative_1577_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1579<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1574_0, Constant_39_0,
// DepthwiseConv2dNative_1579_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1578<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1573_0, Constant_196_0,
// DepthwiseConv2dNative_1578_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1577 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1556
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1579 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1555
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1578 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1555

// Node name:	Add_1554
// Description:	Add
// Input:
//	- name: AvgPool_1548_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1480_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Output:
//	- name: Add_1554_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_1554_block_kernel(float *input0, float *input1,
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
// Node name:	DepthwiseConv2dNative_1555
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1549_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_135_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1555_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1555_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1556
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1549_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_338_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1556_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1556_block_kernel(
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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_152(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    Add_float_float_float_cuda_Add_1554_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 79) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1555_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 16, NULL);
  } else if ((int)blockIdx.x >= 80 && (int)blockIdx.x <= 143) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1556_block_kernel(
        input2, input4, output2, threadIdx.x, blockIdx.x - 80, NULL);
  } else if ((int)blockIdx.x >= 144 && (int)blockIdx.x <= 207) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1556_block_kernel(
        input5, input6, output3, threadIdx.x, blockIdx.x - 144, NULL);
  } else if ((int)blockIdx.x >= 208 && (int)blockIdx.x <= 271) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1555_block_kernel(
        input7, input8, output4, threadIdx.x, blockIdx.x - 208, NULL);
  } else if ((int)blockIdx.x >= 272 && (int)blockIdx.x <= 335) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1555_block_kernel(
        input9, input10, output5, threadIdx.x, blockIdx.x - 272, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_152_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_152<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4, output5);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1608_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_232_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Constant_312_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Constant_333_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: BatchNormInference_1607_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Output:
//	- name: DepthwiseConv2dNative_1613_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1611_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1612_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Slice_1610_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1613<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1608_0, Constant_232_0,
// DepthwiseConv2dNative_1613_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1611<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1608_0, Constant_312_0,
// DepthwiseConv2dNative_1611_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1612<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1608_0, Constant_333_0,
// DepthwiseConv2dNative_1612_0); Slice_float_float_cuda_Slice_1610<<<dim3(128,
// 1, 1), dim3(64, 1, 1), 0, 0>>>(BatchNormInference_1607_0, Slice_1610_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1612 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1611

// Node name:	DepthwiseConv2dNative_1613
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1608_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_232_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1613_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1613_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1611
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1608_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_312_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1611_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1611_block_kernel(
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
// Node name:	Slice_1610
// Description:	Slice
// Input:
//	- name: BatchNormInference_1607_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Output:
//	- name: Slice_1610_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Slice_float_float_cuda_Slice_1610_block_kernel(float *input0, float *output0,
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_159(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1613_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1611_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 64, NULL);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1611_block_kernel(
        input0, input3, output2, threadIdx.x, blockIdx.x - 128, NULL);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 319) {
    Slice_float_float_cuda_Slice_1610_block_kernel(input4, output3, threadIdx.x,
                                                   blockIdx.x - 192, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_159_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_159<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    output0, output1, output2, output3);
}
