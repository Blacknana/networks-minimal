// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_2830
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2830_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2830(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2830_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2830_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3146
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3146_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3146(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3146_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3146_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2059
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2059_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2059(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2059_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2059_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_179
// Description:	Constant
// Input:
// Output:
//	- name: Constant_179_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_179(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_179_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_179_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2868
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2868_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2868(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2868_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2868_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2925
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2925_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2925(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2925_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2925_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_309
// Description:	Constant
// Input:
// Output:
//	- name: Constant_309_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_309(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_309_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_309_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_89
// Description:	Constant
// Input:
// Output:
//	- name: Constant_89_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_89(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_89_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_89_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2837
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2837_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2837(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2837_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2837_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3022
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3022_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_3022(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3022_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3022_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3078
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3078_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3078(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3078_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3078_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1097_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_368_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Constant_321_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: AvgPool_1098_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1029_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Relu_1123_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_200_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Relu_1121_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_230_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Relu_1122_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_78_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1103_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: DepthwiseConv2dNative_1104_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Add_1105_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1128_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: DepthwiseConv2dNative_1126_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: DepthwiseConv2dNative_1127_0	type: float	shape: Shape{1,
//64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1103<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1097_0, Constant_368_0,
// DepthwiseConv2dNative_1103_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1097_0, Constant_321_0,
// DepthwiseConv2dNative_1104_0); Add_float_float_float_cuda_Add_1105<<<dim3(32,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1098_0, BatchNormInference_1029_0,
// Add_1105_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1128<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1123_0, Constant_200_0,
// DepthwiseConv2dNative_1128_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1126<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1121_0, Constant_230_0,
// DepthwiseConv2dNative_1126_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1127<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1122_0, Constant_78_0,
// DepthwiseConv2dNative_1127_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1128 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1126 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1103
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1127 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104

// Node name:	DepthwiseConv2dNative_1103
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1097_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_368_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1103_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1103_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1104
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1097_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_321_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1104_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104_block_kernel(
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
// Node name:	Add_1105
// Description:	Add
// Input:
//	- name: AvgPool_1098_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1029_0	type: float	shape: Shape{1,
//64, 16, 16}
// Output:
//	- name: Add_1105_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_1105_block_kernel(float *input0, float *input1,
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_87(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_1105_block_kernel(
        input3, input4, output2, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 159) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1103_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 32, NULL);
  } else if ((int)blockIdx.x >= 160 && (int)blockIdx.x <= 287) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 160, NULL);
  } else if ((int)blockIdx.x >= 288 && (int)blockIdx.x <= 415) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104_block_kernel(
        input5, input6, output3, threadIdx.x, blockIdx.x - 288, NULL);
  } else if ((int)blockIdx.x >= 416 && (int)blockIdx.x <= 543) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1103_block_kernel(
        input7, input8, output4, threadIdx.x, blockIdx.x - 416, NULL);
  } else if ((int)blockIdx.x >= 544 && (int)blockIdx.x <= 671) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104_block_kernel(
        input9, input10, output5, threadIdx.x, blockIdx.x - 544, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_87_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_87<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4, output5);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1308_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_221_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Relu_1306_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_169_0	type: float	shape: Shape{7, 7, 128, 1}
//	- name: Relu_1307_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_258_0	type: float	shape: Shape{7, 7, 128, 1}
//	- name: Slice_1284_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: DepthwiseConv2dNative_1313_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: DepthwiseConv2dNative_1311_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: DepthwiseConv2dNative_1312_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Relu_1288_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1313<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1308_0, Constant_221_0,
// DepthwiseConv2dNative_1313_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1311<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1306_0, Constant_169_0,
// DepthwiseConv2dNative_1311_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1312<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1307_0, Constant_258_0,
// DepthwiseConv2dNative_1312_0); Relu_float_float_cuda_Relu_1288<<<dim3(32, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Slice_1284_0, Relu_1288_0); Deduped function
// map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1312 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1311

// Node name:	DepthwiseConv2dNative_1313
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1308_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_221_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1313_0	type: float	shape: Shape{1,
//128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1313_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1311
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1306_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_169_0	type: float	shape: Shape{7, 7, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1311_0	type: float	shape: Shape{1,
//128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1311_block_kernel(
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
  const int filter_height = 7;
  const int filter_width = 7;
  const int depth_multiplier = 1;
  const int stride = 1;
  const int pad_height = 3;
  const int pad_width = 3;
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
// Node name:	Relu_1288
// Description:	Relu
// Input:
//	- name: Slice_1284_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1288_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Relu_float_float_cuda_Relu_1288_block_kernel(float *input0, float *output0,
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Relu_113(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *output0, float *output1,
    float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Relu_float_float_cuda_Relu_1288_block_kernel(input6, output3, threadIdx.x,
                                                 blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 95) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1313_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 32, NULL);
  } else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 159) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1311_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 96, NULL);
  } else if ((int)blockIdx.x >= 160 && (int)blockIdx.x <= 223) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1311_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 160, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Relu_113_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *output0, float *output1,
    float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Relu_113<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, output0, output1, output2,
                                    output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1423_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2548_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3130_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1425_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2554_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3134_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1424_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2551_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3132_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: AvgPool_1419_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Slice_1421_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1444_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1446_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1445_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Add_1422_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1426_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3129<<<dim3(1,
// 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1423_0, Constant_2548_0,
// Constant_3130_0, Relu_1444_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3133<<<dim3(1,
// 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1425_0, Constant_2554_0,
// Constant_3134_0, Relu_1446_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3131<<<dim3(1,
// 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1424_0, Constant_2551_0,
// Constant_3132_0, Relu_1445_0); Add_float_float_float_cuda_Add_1422<<<dim3(16,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1419_0, AvgPool_1419_0, Add_1422_0);
// Relu_float_float_cuda_Relu_1426<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Slice_1421_0, Relu_1426_0); Deduped function map: <src_function_name :
// deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3133 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3129
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3131 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3129

// Node name:	Matched_Pattern_3129
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1423_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2548_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3130_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1444_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3129_block_kernel(
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
// Node name:	Add_1422
// Description:	Add
// Input:
//	- name: AvgPool_1419_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: AvgPool_1419_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1422_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_1422_block_kernel(float *input0, float *input1,
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
// Node name:	Relu_1426
// Description:	Relu
// Input:
//	- name: Slice_1421_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1426_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Relu_float_float_cuda_Relu_1426_block_kernel(float *input0, float *output0,
                                             int thread_id, int block_id,
                                             char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(16, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  output0[blockIdx.x * 512 + threadIdx.x] =
      relu(input0[blockIdx.x * 512 + threadIdx.x]);
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_Relu_133(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {

  __shared__ char shared_buffer[1536];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    Add_float_float_float_cuda_Add_1422_block_kernel(
        input9, input9, output3, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31) {
    Relu_float_float_cuda_Relu_1426_block_kernel(
        input10, output4, threadIdx.x, blockIdx.x - 16, shared_buffer);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 95) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3129_block_kernel(
        input0, input1, input2, output0, threadIdx.x, blockIdx.x - 32,
        shared_buffer);
  } else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 159) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3129_block_kernel(
        input3, input4, input5, output1, threadIdx.x, blockIdx.x - 96,
        shared_buffer);
  } else if ((int)blockIdx.x >= 160 && (int)blockIdx.x <= 223) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3129_block_kernel(
        input6, input7, input8, output2, threadIdx.x, blockIdx.x - 160,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_Relu_133_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_Relu_133<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4);
}
// Node name:	Dot_1760
// Description:	Dot
// Input:
//	- name: Divide_1759_0	type: float	shape: Shape{1, 768}
//	- name: Constant_422_0	type: float	shape: Shape{768, 10}
// Output:
//	- name: Dot_1760_0	type: float	shape: Shape{1, 10}
void Dot_float_float_float_cuda_lib_Dot_1760(cublasHandle_t cublas_handle,
                                             float *input0, float *input1,
                                             float *output0) {
  const float alpha = 1.0;
  const float beta = 0;
  CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 10, 1,
                               768, &alpha, static_cast<const float *>(input1),
                               10, static_cast<const float *>(input0), 768,
                               &beta, static_cast<float *>(output0), 10));
}
