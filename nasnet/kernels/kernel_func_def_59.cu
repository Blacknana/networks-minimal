// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_2990
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2990_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2990(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2990_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2990_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3112
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3112_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3112(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3112_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3112_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3154
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3154_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3154(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3154_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3154_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3086
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3086_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3086(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3086_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3086_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_283
// Description:	Constant
// Input:
// Output:
//	- name: Constant_283_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_283(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_283_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_283_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2820
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2820_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2820(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2820_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2820_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2281
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2281_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2281(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2281_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2281_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_290
// Description:	Constant
// Input:
// Output:
//	- name: Constant_290_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_290(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_290_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_290_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2197
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2197_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2197(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2197_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2197_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3070
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3070_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3070(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3070_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3070_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: BatchNormInference_1218_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Relu_1221_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_167_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Constant_98_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Constant_96_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: Slice_1220_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1227_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: DepthwiseConv2dNative_1225_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: DepthwiseConv2dNative_1226_0	type: float	shape: Shape{1,
//64, 16, 16}
// Fused functions:
// Slice_float_float_cuda_Slice_1220<<<dim3(256, 1, 1), dim3(64, 1, 1), 0,
// 0>>>(BatchNormInference_1218_0, Slice_1220_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1227<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1221_0, Constant_167_0,
// DepthwiseConv2dNative_1227_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1225<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1221_0, Constant_98_0,
// DepthwiseConv2dNative_1225_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1226<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1221_0, Constant_96_0,
// DepthwiseConv2dNative_1226_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1225 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1227

// Node name:	Slice_1220
// Description:	Slice
// Input:
//	- name: BatchNormInference_1218_0	type: float	shape: Shape{1,
//64, 16, 16}
// Output:
//	- name: Slice_1220_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Slice_float_float_cuda_Slice_1220_block_kernel(float *input0, float *output0,
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
// Node name:	DepthwiseConv2dNative_1227
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1221_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_167_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1227_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1227_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1226
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1221_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_96_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1226_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1226_block_kernel(
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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_Slice_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_103(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1227_block_kernel(
        input1, input2, output1, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 255) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1227_block_kernel(
        input1, input3, output2, threadIdx.x, blockIdx.x - 128, NULL);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 383) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1226_block_kernel(
        input1, input4, output3, threadIdx.x, blockIdx.x - 256, NULL);
  } else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 639) {
    Slice_float_float_cuda_Slice_1220_block_kernel(input0, output0, threadIdx.x,
                                                   blockIdx.x - 384, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_Slice_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_103_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_Slice_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_103<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    output0, output1, output2, output3);
}
// Node name:	AvgPool_1352
// Description:	AvgPool
// Input:
//	- name: Slice_1347_0	type: float	shape: Shape{1, 128, 16, 16}
// Output:
//	- name: AvgPool_1352_0	type: float	shape: Shape{1, 128, 8, 8}
void AvgPool_float_float_cuda_lib_AvgPool_1352(cudnnHandle_t cudnn_handle,
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
      desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
      CUDNN_NOT_PROPAGATE_NAN, 3, 3, 0, 0, 2, 2));
  const float alpha = 1.0;
  const float beta = 0.0;
  CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc,
                                      input0, &beta, output_desc, output0));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
  CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));
}
// Node name:	AvgPool_1352
// Description:	AvgPool
// Input:
//	- name: Slice_1347_0	type: float	shape: Shape{1, 128, 16, 16}
// Output:
//	- name: AvgPool_1352_0	type: float	shape: Shape{1, 128, 8, 8}
// 3, 0, 2(kernelH, pad, stride)
// grid(32,1,1) block(256,1,1)
__device__ void operator_avg_pool_h_192_16_16_3x3_2(const float *input,
                                                    float *output,
                                                    int blockidx) {

  const int pooled_height = 8;
  const int pooled_width = 8;
  const int nthreads = 8192;
  int index = blockidx * 256 + threadIdx.x;

  if (index < nthreads) {
    const int kChannels = 128;
    const int kHeight = 16;
    const int kWidth = 16;
    const int kKernelH = 3;
    const int kKernelW = 3;
    const int kPadH = 0;
    const int kPadW = 0;
    const int kStrideH = 2;
    const int kStrideW = 2;

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
        avgval = (input[slice_offset + h * kWidth + w]) /
                     ((hend - hstart) * (wend - wstart)) +
                 avgval;
      }
    }

    // output
    output[index] = avgval;
  }
}

// Node name:	MaxPool_1351
// Description:	MaxPool
// Input:
//	- name: Slice_1347_0	type: float	shape: Shape{1, 128, 16, 16}
// Output:
//	- name: MaxPool_1351_0	type: float	shape: Shape{1, 128, 8, 8}
// 3, 0, 2(kernelH, pad, stride)
// grid(32,1,1) block(256,1,1)
__device__ void operator_max_pool_h_128_16_16_3x3_2(const float *input,
                                                    float *output,
                                                    int blockidx) {

  const int pooled_height = 8;
  const int pooled_width = 8;
  const int nthreads = 8192;
  int index = blockidx * 256 + threadIdx.x;

  if (index < nthreads) {
    const int kChannels = 128;
    const int kHeight = 16;
    const int kWidth = 16;
    const int kKernelH = 3;
    const int kKernelW = 3;
    const int kPadH = 0;
    const int kPadW = 0;
    const int kStrideH = 2;
    const int kStrideW = 2;

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

    // get max value postion
    float maxval = -FLT_MAX;
    float tmp;
    int slice_offset = (n * kChannels + c) * kHeight * kWidth;

#pragma unroll 4
    for (int h = hstart; h < hend; ++h) {
#pragma unroll 4
      for (int w = wstart; w < wend; ++w) {
        tmp = input[slice_offset + h * kWidth + w];
        if (tmp > maxval) {
          maxval = tmp;
        }
      }
    }
    // output
    output[index] = maxval;
  }
}

// Node name:	AvgPool_1337
// Description:	AvgPool
// Input:
//	- name: Relu_1336_0	type: float	shape: Shape{1, 384, 16, 16}
// Output:
//	- name: AvgPool_1337_0	type: float	shape: Shape{1, 384, 8, 8}
// 1, 0, 2(kernelH, pad, stride)
// grid(96,1,1) block(256,1,1)
__device__ void operator_avg_pool_h_384_16_16_1x1_2(const float *input,
                                                    float *output,
                                                    int blockidx) {

  const int pooled_height = 8;
  const int pooled_width = 8;
  const int nthreads = 24576;
  int index = blockidx * 256 + threadIdx.x;

  if (index < nthreads) {
    const int kChannels = 384;
    const int kHeight = 16;
    const int kWidth = 16;
    const int kKernelH = 1;
    const int kKernelW = 1;
    const int kPadH = 0;
    const int kPadW = 0;
    const int kStrideH = 2;
    const int kStrideW = 2;

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
        avgval = (input[slice_offset + h * kWidth + w]) /
                     ((hend - hstart) * (wend - wstart)) +
                 avgval;
      }
    }

    // output
    output[index] = avgval;
  }
}

extern "C" __global__ void
BlockFusionKernel_2_AvgPool1352_MaxPool1351_AvgPool1337(
    float *input0, float *input1, float *input2, float *output0, float *output1,
    float *output2) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    operator_avg_pool_h_192_16_16_3x3_2(input0, output0, blockIdx.x);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    operator_max_pool_h_128_16_16_3x3_2(input1, output1, blockIdx.x - 32);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 159) {
    operator_avg_pool_h_384_16_16_1x1_2(input2, output2, blockIdx.x - 64);
  }
}

extern void BlockFusionKernel_2_AvgPool1352_MaxPool1351_AvgPool1337_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0, float *output1,
    float *output2) {
  BlockFusionKernel_2_AvgPool1352_MaxPool1351_AvgPool1337<<<grids, blocks, mem,
                                                            stream>>>(
      input0, input1, input2, output0, output1, output2);
}
// Node name:	Convolution_1408
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1406_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2539_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1408_0	type: float	shape: Shape{1, 128, 8,
//8}
extern "C" __global__ void Convolution_float_float_float_cuda_Convolution_1408(
    float *input0, float *input1, float *output0) {
  __shared__ float pad_temp_shared[256];
  __shared__ float input1_shared[128];
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
extern void Convolution_float_float_float_cuda_Convolution_1408_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *output0) {
  Convolution_float_float_float_cuda_Convolution_1408<<<grids, blocks, mem,
                                                        stream>>>(
      input0, input1, output0);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_736_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2173_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_738_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2179_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_737_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2176_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_721_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2167_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3022_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_722_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2170_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3024_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Convolution_742_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Convolution_746_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Convolution_744_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Relu_739_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_740_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_742<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_736_0, Constant_2173_0,
// Convolution_742_0);
// Convolution_float_float_float_cuda_Convolution_746<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_738_0, Constant_2179_0,
// Convolution_746_0);
// Convolution_float_float_float_cuda_Convolution_744<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_737_0, Constant_2176_0,
// Convolution_744_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3021<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_721_0, Constant_2167_0,
// Constant_3022_0, Relu_739_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3023<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_722_0, Constant_2170_0,
// Constant_3024_0, Relu_740_0); Deduped function map: <src_function_name :
// deduped_function_name> Convolution_float_float_float_cuda_Convolution_746 :
// Convolution_float_float_float_cuda_Convolution_742
// Convolution_float_float_float_cuda_Convolution_744 :
// Convolution_float_float_float_cuda_Convolution_742
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3023 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3021

// Node name:	Convolution_742
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_736_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2173_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_742_0	type: float	shape: Shape{1, 32, 32,
//32}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_742_block_kernel(
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
// Node name:	Matched_Pattern_3021
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_721_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2167_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3022_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_739_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3021_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Matched_Pattern_Matched_Pattern_32(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {

  __shared__ char shared_buffer[3072];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_742_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_742_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    Convolution_float_float_float_cuda_Convolution_742_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 128, shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3021_block_kernel(
        input6, input7, input8, output3, threadIdx.x, blockIdx.x - 192,
        shared_buffer);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3021_block_kernel(
        input9, input10, input11, output4, threadIdx.x, blockIdx.x - 256,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Matched_Pattern_Matched_Pattern_32_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Matched_Pattern_Matched_Pattern_32<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, input11, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1313_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2500_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1311_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2494_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1312_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2497_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Relu_1288_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_359_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Constant_282_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: AvgPool_1289_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1219_0	type: float	shape: Shape{1,
//64, 16, 16}
// Output:
//	- name: Convolution_1321_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Convolution_1317_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Convolution_1319_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: DepthwiseConv2dNative_1296_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: DepthwiseConv2dNative_1297_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Add_1298_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1321<<<dim3(1, 4, 8), dim3(4,
// 2, 16), 0, 0>>>(DepthwiseConv2dNative_1313_0, Constant_2500_0,
// Convolution_1321_0);
// Convolution_float_float_float_cuda_Convolution_1317<<<dim3(1, 4, 8), dim3(4,
// 2, 16), 0, 0>>>(DepthwiseConv2dNative_1311_0, Constant_2494_0,
// Convolution_1317_0);
// Convolution_float_float_float_cuda_Convolution_1319<<<dim3(1, 4, 8), dim3(4,
// 2, 16), 0, 0>>>(DepthwiseConv2dNative_1312_0, Constant_2497_0,
// Convolution_1319_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1296<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1288_0, Constant_359_0,
// DepthwiseConv2dNative_1296_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1297<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1288_0, Constant_282_0,
// DepthwiseConv2dNative_1297_0); Add_float_float_float_cuda_Add_1298<<<dim3(32,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1289_0, BatchNormInference_1219_0,
// Add_1298_0); Deduped function map: <src_function_name :
// deduped_function_name> Convolution_float_float_float_cuda_Convolution_1317 :
// Convolution_float_float_float_cuda_Convolution_1321
// Convolution_float_float_float_cuda_Convolution_1319 :
// Convolution_float_float_float_cuda_Convolution_1321

// Node name:	Convolution_1321
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1313_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2500_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1321_0	type: float	shape: Shape{1, 128, 8,
//8}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1321_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(4, 2, 16);
  const dim3 gridDim(1, 4, 8);
  const dim3 threadIdx(thread_id % 4, thread_id / 4 % 2, thread_id / 8);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 4, block_id / 4);
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
          input0[((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                   (((int)threadIdx.y) * 8)) +
                  (((int)threadIdx.x) * 2))];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                   (((int)threadIdx.y) * 8)) +
                  (((int)threadIdx.x) * 2))];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                                (((int)threadIdx.x) * 2))] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  1024)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  1025)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  16)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  17)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                                (((int)threadIdx.x) * 2))] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  2048)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  2049)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  32)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  33)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                                (((int)threadIdx.x) * 2))] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  3072)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  3073)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  48)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  49)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                                (((int)threadIdx.x) * 2))] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  4096)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  4097)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  64)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  65)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                                (((int)threadIdx.x) * 2))] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  5120)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  5121)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  80)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  81)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                                (((int)threadIdx.x) * 2))] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  6144)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  6145)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  96)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  97)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                                (((int)threadIdx.x) * 2))] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  7168)];
      pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          input0[(((((((int)threadIdx.z) * 64) + (((int)blockIdx.y) * 16)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  7169)];
      input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  112)];
      input1_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  113)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) +
                                                (((int)threadIdx.x) * 2))] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1)] *
            input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17)] *
            input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33)] *
            input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49)] *
            input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65)] *
            input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81)] *
            input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97)] *
            input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113)] *
            input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129)] *
            input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145)] *
            input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161)] *
            input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177)] *
            input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193)] *
            input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209)] *
            input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225)] *
            input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] =
          (compute_local[1] +
           (pad_temp_shared[(
                ((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241)] *
            input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                 (((int)blockIdx.y) * 16)) +
                (((int)threadIdx.y) * 8)) +
               (((int)threadIdx.x) * 2))] = compute_local[0];
      compute[((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                  (((int)blockIdx.y) * 16)) +
                 (((int)threadIdx.y) * 8)) +
                (((int)threadIdx.x) * 2)) +
               1)] = compute_local[1];
    }
  }
}
// Node name:	DepthwiseConv2dNative_1296
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1288_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_359_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1296_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1296_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1297
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1288_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_282_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1297_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1297_block_kernel(
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
// Node name:	Add_1298
// Description:	Add
// Input:
//	- name: AvgPool_1289_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1219_0	type: float	shape: Shape{1,
//64, 16, 16}
// Output:
//	- name: Add_1298_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_1298_block_kernel(float *input0, float *input1,
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

__device__ __forceinline__ void
Fuse_AvgPool_911_Add_1298(float *input0, float *input1, float *output0,
                          int thread_id, int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }

  const int pooled_height = 16;
  const int pooled_width = 16;
  const int nthreads = 16384;
  int index = block_id * 512 + threadIdx.x;

  if (index < nthreads) {
    const int kChannels = 64;
    const int kHeight = 16;
    const int kWidth = 16;
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
    output0[index] = add(avgval, input1[index]);
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_114(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {

  __shared__ char shared_buffer[2048];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Fuse_AvgPool_911_Add_1298(input9, input10, output5, threadIdx.x,
                              blockIdx.x - 0, shared_buffer);
    // Add_float_float_float_cuda_Add_1298_block_kernel(input9, input10,
    // output5, threadIdx.x, blockIdx.x - 352 + 0, shared_buffer);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_1321_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 32, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 95) {
    Convolution_float_float_float_cuda_Convolution_1321_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  } else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_1321_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 96, shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 255) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1296_block_kernel(
        input6, input7, output3, threadIdx.x, blockIdx.x - 128, shared_buffer);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 383) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1297_block_kernel(
        input6, input8, output4, threadIdx.x, blockIdx.x - 256, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_114_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_114<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4, output5);
}
