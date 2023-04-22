// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_2287
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2287_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2287(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2287_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2287_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_422
// Description:	Constant
// Input:
// Output:
//	- name: Constant_422_0	type: float	shape: Shape{768, 10}
void Constant_float_cuda_Constant_422(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_422_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_422_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[30720];
  bin_file.read(tmp_mem, 30720);
  cudaMemcpyAsync(output0, tmp_mem, 30720, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2680
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2680_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2680(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2680_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2680_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_155
// Description:	Constant
// Input:
// Output:
//	- name: Constant_155_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_155(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_155_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_155_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[3200];
  bin_file.read(tmp_mem, 3200);
  cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2750
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2750_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2750(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2750_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2750_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_88
// Description:	Constant
// Input:
// Output:
//	- name: Constant_88_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_88(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_88_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_88_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2095
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2095_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2095(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2095_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2095_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_245
// Description:	Constant
// Input:
// Output:
//	- name: Constant_245_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_245(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_245_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_245_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[3200];
  bin_file.read(tmp_mem, 3200);
  cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2209
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2209_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2209(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2209_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2209_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2776
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2776_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2776(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2776_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2776_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3138
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3138_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3138(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3138_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3138_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1743_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_206_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Relu_1744_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_384_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1745_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: DepthwiseConv2dNative_1746_0	type: float	shape: Shape{1,
//128, 8, 8}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1745<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1743_0, Constant_206_0,
// DepthwiseConv2dNative_1745_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1746<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1744_0, Constant_384_0,
// DepthwiseConv2dNative_1746_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	DepthwiseConv2dNative_1745
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1743_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_206_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1745_0	type: float	shape: Shape{1,
//128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1745_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1746
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1744_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_384_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1746_0	type: float	shape: Shape{1,
//128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1746_block_kernel(
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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_177(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1745_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1746_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_177_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_177<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_471_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2972_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_480_0	type: float	shape: Shape{1,
//96, 32, 32}
//	- name: Constant_2038_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Constant_2984_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_479_0	type: float	shape: Shape{1,
//96, 32, 32}
//	- name: Constant_2035_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Constant_2982_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_476_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2894_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_474_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2737_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_472_0	type: float	shape: Shape{1,
//96, 32, 32}
//	- name: Constant_2032_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Constant_2980_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_469_0	type: float	shape: Shape{1, 96, 32, 32}
//	- name: Constant_2017_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Constant_2020_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Convolution_478_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2739_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_485_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Relu_512_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_511_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_489_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: BatchNormInference_488_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Relu_498_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_504_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_482_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Convolution_484_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: BatchNormInference_490_0	type: float	shape: Shape{1,
//32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2016<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_471_0, Constant_2972_0, BatchNormInference_485_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983<<<dim3(1,
// 32, 2), dim3(16, 1, 8), 0, 0>>>(DepthwiseConv2dNative_480_0, Constant_2038_0,
// Constant_2984_0, Relu_512_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2981<<<dim3(1,
// 32, 2), dim3(16, 1, 8), 0, 0>>>(DepthwiseConv2dNative_479_0, Constant_2035_0,
// Constant_2982_0, Relu_511_0); Add_float_float_float_cuda_Add_2031<<<dim3(64,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_476_0, Constant_2894_0,
// BatchNormInference_489_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_3<<<dim3(64, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_474_0, Constant_2737_0, Relu_498_0,
// BatchNormInference_488_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2979<<<dim3(1,
// 32, 2), dim3(16, 1, 8), 0, 0>>>(DepthwiseConv2dNative_472_0, Constant_2032_0,
// Constant_2980_0, Relu_504_0);
// Convolution_float_float_float_cuda_Convolution_482<<<dim3(1, 32, 2), dim3(16,
// 1, 8), 0, 0>>>(AvgPool_469_0, Constant_2017_0, Convolution_482_0);
// Convolution_float_float_float_cuda_Convolution_484<<<dim3(1, 32, 2), dim3(16,
// 1, 8), 0, 0>>>(AvgPool_469_0, Constant_2020_0, Convolution_484_0);
// Add_float_float_float_cuda_Add_2025<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_478_0, Constant_2739_0, BatchNormInference_490_0); Deduped
// function map: <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2981 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983
// Add_float_float_float_cuda_Add_2031 : Add_float_float_float_cuda_Add_2016
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2979 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983
// Convolution_float_float_float_cuda_Convolution_484 :
// Convolution_float_float_float_cuda_Convolution_482
// Add_float_float_float_cuda_Add_2025 : Add_float_float_float_cuda_Add_2016

// Node name:	Add_2016
// Description:	Add
// Input:
//	- name: Convolution_471_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2972_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_485_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2016_block_kernel(float *input0, float *input1,
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
// Node name:	Matched_Pattern_2983
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_480_0	type: float	shape: Shape{1,
//96, 32, 32}
//	- name: Constant_2038_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Constant_2984_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_512_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983_block_kernel(
    float *input0, float *input1, float *input2, float *output0, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(16, 1, 8);
  const dim3 gridDim(1, 32, 2);
  const dim3 threadIdx(thread_id % 16, 0, thread_id / 16);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 32, block_id / 32);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 2048);
  {
    float *compute = output0;
    {
      float compute1[4];

#pragma unroll
      for (int xx_init = 0; xx_init < 2; ++xx_init) {
        compute1[xx_init] = 0.000000e+00f;
        compute1[(xx_init + 2)] = 0.000000e+00f;
      }
#pragma unroll
      for (int rc_outer = 0; rc_outer < 6; ++rc_outer) {
        __syncthreads();
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
          pad_temp_shared[(
              ((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 4)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] =
              input0[(
                  ((((rc_outer * 16384) + (((int)threadIdx.z) * 2048)) +
                    ((((((int)threadIdx.x) * 4) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >>
                      5) *
                     1024)) +
                   (((int)blockIdx.y) * 32)) +
                  (((((int)threadIdx.x) * 4) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) &
                   31))];
        }
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 2;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
          input1_shared[(
              ((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 2)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] =
              input1[(
                  ((((((int)blockIdx.z) * 1536) + (((int)threadIdx.z) * 192)) +
                    ((((((int)threadIdx.x) * 2) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >>
                      4) *
                     96)) +
                   (rc_outer * 16)) +
                  (((((int)threadIdx.x) * 2) +
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
                                 128)]));
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
            8192)] =
            max((compute1[(i3_inner_inner_inner + 2)] +
                 input2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8)]),
                0.000000e+00f);
      }
    }
  }
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_474_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2737_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_498_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_488_0	type: float	shape: Shape{1,
//32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2028<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_474_0, Constant_2737_0, BatchNormInference_488_0);
// Relu_float_float_cuda_Relu_498<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_488_0, Relu_498_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Relu_3_block_kernel(
    float *input0, float *input1, float *output0, float *output1, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(64, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = relu(temp0);
  output1[tid] = temp0;
  output0[tid] = temp1;
}
// Node name:	Convolution_482
// Description:	Convolution
// Input:
//	- name: AvgPool_469_0	type: float	shape: Shape{1, 96, 32, 32}
//	- name: Constant_2017_0	type: float	shape: Shape{32, 96, 1, 1}
// Output:
//	- name: Convolution_482_0	type: float	shape: Shape{1, 32, 32,
//32}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_482_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(16, 1, 8);
  const dim3 gridDim(1, 32, 2);
  const dim3 threadIdx(thread_id % 16, 0, thread_id / 16);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 32, block_id / 32);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 2048);
  {
    float *compute = output0;
    {
      float compute_local[4];

#pragma unroll
      for (int xx_c_init = 0; xx_c_init < 2; ++xx_c_init) {
        compute_local[xx_c_init] = 0.000000e+00f;
        compute_local[(xx_c_init + 2)] = 0.000000e+00f;
      }
#pragma unroll
      for (int rc_outer = 0; rc_outer < 6; ++rc_outer) {
        __syncthreads();
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
          pad_temp_shared[(
              ((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 4)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] =
              input0[(
                  ((((rc_outer * 16384) + (((int)threadIdx.z) * 2048)) +
                    ((((((int)threadIdx.x) * 4) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >>
                      5) *
                     1024)) +
                   (((int)blockIdx.y) * 32)) +
                  (((((int)threadIdx.x) * 4) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) &
                   31))];
        }
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 2;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
          input1_shared[(
              ((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 2)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] =
              input1[(
                  ((((((int)blockIdx.z) * 1536) + (((int)threadIdx.z) * 192)) +
                    ((((((int)threadIdx.x) * 2) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >>
                      4) *
                     96)) +
                   (rc_outer * 16)) +
                  (((((int)threadIdx.x) * 2) +
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
                                 128)]));
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
            8192)] = compute_local[(xx_inner_inner_inner + 2)];
      }
    }
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Matched_Pattern_Matched_Pattern_Add_fused_kernel_Matched_Pattern_Convolution_Convolution_Add_1(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *input12, float *input13,
    float *input14, float *input15, float *input16, float *input17,
    float *input18, float *input19, float *output0, float *output1,
    float *output2, float *output3, float *output4, float *output5,
    float *output6, float *output7, float *output8, float *output9) {

  __shared__ char shared_buffer[3072];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Add_float_float_float_cuda_Add_2016_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Add_float_float_float_cuda_Add_2016_block_kernel(
        input8, input9, output3, threadIdx.x, blockIdx.x - 64, shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    FusedKernel_float_float_float_float_cuda_Add_Relu_3_block_kernel(
        input10, input11, output5, output4, threadIdx.x, blockIdx.x - 128,
        shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    Add_float_float_float_cuda_Add_2016_block_kernel(
        input18, input19, output9, threadIdx.x, blockIdx.x - 192,
        shared_buffer);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983_block_kernel(
        input2, input3, input4, output1, threadIdx.x, blockIdx.x - 256,
        shared_buffer);
  } else if ((int)blockIdx.x >= 320 && (int)blockIdx.x <= 383) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983_block_kernel(
        input5, input6, input7, output2, threadIdx.x, blockIdx.x - 320,
        shared_buffer);
  } else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 447) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983_block_kernel(
        input12, input13, input14, output6, threadIdx.x, blockIdx.x - 384,
        shared_buffer);
  } else if ((int)blockIdx.x >= 448 && (int)blockIdx.x <= 511) {
    Convolution_float_float_float_cuda_Convolution_482_block_kernel(
        input15, input16, output7, threadIdx.x, blockIdx.x - 448,
        shared_buffer);
  } else if ((int)blockIdx.x >= 512 && (int)blockIdx.x <= 575) {
    Convolution_float_float_float_cuda_Convolution_482_block_kernel(
        input15, input17, output8, threadIdx.x, blockIdx.x - 512,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Matched_Pattern_Matched_Pattern_Add_fused_kernel_Matched_Pattern_Convolution_Convolution_Add_1_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *input12, float *input13,
    float *input14, float *input15, float *input16, float *input17,
    float *input18, float *input19, float *output0, float *output1,
    float *output2, float *output3, float *output4, float *output5,
    float *output6, float *output7, float *output8, float *output9) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Matched_Pattern_Matched_Pattern_Add_fused_kernel_Matched_Pattern_Convolution_Convolution_Add_1<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, input11, input12, input13, input14, input15, input16,
      input17, input18, input19, output0, output1, output2, output3, output4,
      output5, output6, output7, output8, output9);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_904_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2263_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3048_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_76_0	type: float	shape: Shape{64}
//	- name: Constant_53_0	type: float	shape: Shape{64}
//	- name: Concat_905_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_912_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_908_0	type: float	shape: Shape{1,
//64, 16, 16}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3047<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_904_0, Constant_2263_0,
// Constant_3048_0, Relu_912_0);
// BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_908<<<dim3(64,
// 1, 1), dim3(256, 1, 1), 0, 0>>>(Constant_76_0, Constant_53_0, Concat_905_0,
// Constant_53_0, Constant_76_0, BatchNormInference_908_0); Deduped function
// map: <src_function_name : deduped_function_name>

// Node name:	Matched_Pattern_3047
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_904_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2263_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3048_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_912_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3047_block_kernel(
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
// Node name:	BatchNormInference_908
// Description:	BatchNormInference
// Input:
//	- name: Constant_76_0	type: float	shape: Shape{64}
//	- name: Constant_53_0	type: float	shape: Shape{64}
//	- name: Concat_905_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_53_0	type: float	shape: Shape{64}
//	- name: Constant_76_0	type: float	shape: Shape{64}
// Output:
//	- name: BatchNormInference_908_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ void
BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_908_block_kernel(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, int thread_id, int block_id, char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(256, 1, 1);
  const dim3 gridDim(64, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  const int st = blockIdx.x * 16 * 16;
  const int c_id = blockIdx.x % 64;
#pragma unroll 1
  for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
    output0[st + i] =
        (input1[c_id] + (input0[c_id] * (input2[st + i] - input3[c_id]) /
                         sqrtf(1.001e-05 + input4[c_id])));
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_BatchNormInference_58(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {

  __shared__ char shared_buffer[2048];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    BatchNormInference_float_float_float_float_float_float_cuda_BatchNormInference_908_block_kernel(
        input3, input4, input5, input4, input3, output1, threadIdx.x,
        blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3047_block_kernel(
        input0, input1, input2, output0, threadIdx.x, blockIdx.x - 64,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_BatchNormInference_58_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_BatchNormInference_58<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_596_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2095_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3002_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_597_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2098_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3004_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_612_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2107_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_610_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2101_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_611_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2104_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Relu_613_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_614_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_620_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Convolution_616_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Convolution_618_0	type: float	shape: Shape{1, 32, 32,
//32}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3001<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_596_0, Constant_2095_0,
// Constant_3002_0, Relu_613_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3003<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_597_0, Constant_2098_0,
// Constant_3004_0, Relu_614_0);
// Convolution_float_float_float_cuda_Convolution_620<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_612_0, Constant_2107_0,
// Convolution_620_0);
// Convolution_float_float_float_cuda_Convolution_616<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_610_0, Constant_2101_0,
// Convolution_616_0);
// Convolution_float_float_float_cuda_Convolution_618<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_611_0, Constant_2104_0,
// Convolution_618_0); Deduped function map: <src_function_name :
// deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3003 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3001
// Convolution_float_float_float_cuda_Convolution_616 :
// Convolution_float_float_float_cuda_Convolution_620
// Convolution_float_float_float_cuda_Convolution_618 :
// Convolution_float_float_float_cuda_Convolution_620

// Node name:	Matched_Pattern_3001
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_596_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2095_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3002_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_613_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3001_block_kernel(
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
// Node name:	Convolution_620
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_612_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2107_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_620_0	type: float	shape: Shape{1, 32, 32,
//32}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_620_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_14(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {

  __shared__ char shared_buffer[3072];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3001_block_kernel(
        input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3001_block_kernel(
        input3, input4, input5, output1, threadIdx.x, blockIdx.x - 64,
        shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    Convolution_float_float_float_cuda_Convolution_620_block_kernel(
        input6, input7, output2, threadIdx.x, blockIdx.x - 128, shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    Convolution_float_float_float_cuda_Convolution_620_block_kernel(
        input8, input9, output3, threadIdx.x, blockIdx.x - 192, shared_buffer);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319) {
    Convolution_float_float_float_cuda_Convolution_620_block_kernel(
        input10, input11, output4, threadIdx.x, blockIdx.x - 256,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_14_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_14<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, input11, output0, output1, output2, output3, output4);
}
