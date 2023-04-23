// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_237
// Description:	Constant
// Input:
// Output:
//	- name: Constant_237_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_237(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_237_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_237_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2614
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2614_0	type: float	shape: Shape{128, 768, 1, 1}
void Constant_float_cuda_Constant_2614(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2614_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2614_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[393216];
  bin_file.read(tmp_mem, 393216);
  cudaMemcpyAsync(output0, tmp_mem, 393216, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2629
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2629_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2629(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2629_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2629_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3098
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3098_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3098(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3098_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3098_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_267
// Description:	Constant
// Input:
// Output:
//	- name: Constant_267_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_267(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_267_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_267_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_1853
// Description:	Constant
// Input:
// Output:
//	- name: Constant_1853_0	type: float	shape: Shape{32, 192, 1, 1}
void Constant_float_cuda_Constant_1853(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_1853_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_1853_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[24576];
  bin_file.read(tmp_mem, 24576);
  cudaMemcpyAsync(output0, tmp_mem, 24576, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3140
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3140_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3140(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3140_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3140_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2251
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2251_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2251(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2251_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2251_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_262
// Description:	Constant
// Input:
// Output:
//	- name: Constant_262_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_262(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_262_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_262_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3062
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3062_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3062(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3062_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3062_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2353
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2353_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2353(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2353_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2353_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_998_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_329_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Relu_999_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_403_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Constant_2879_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1006_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2880_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1010_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Convolution_1008_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2864_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: DepthwiseConv2dNative_1003_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: DepthwiseConv2dNative_1004_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Add_1020_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1016_0	type: float	shape: Shape{1,
//64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1003<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_998_0, Constant_329_0,
// DepthwiseConv2dNative_1003_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1004<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_999_0, Constant_403_0,
// DepthwiseConv2dNative_1004_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_29<<<dim3(32, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1006_0, Constant_2879_0,
// Convolution_1010_0, Constant_2880_0, Add_1020_0);
// Add_float_float_float_cuda_Add_2319<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1008_0, Constant_2864_0, BatchNormInference_1016_0); Deduped
// function map: <src_function_name : deduped_function_name>

// Node name:	DepthwiseConv2dNative_1003
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_998_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_329_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1003_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ static void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1003_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1004
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_999_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_403_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1004_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ static void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1004_block_kernel(
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
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1006_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2879_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1010_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2880_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1020_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2316<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1006_0, Constant_2879_0, BatchNormInference_1015_0);
// Add_float_float_float_cuda_Add_2322<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1010_0, Constant_2880_0, BatchNormInference_1017_0);
// Add_float_float_float_cuda_Add_1020<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1017_0, BatchNormInference_1015_0, Add_1020_0);
__device__ __forceinline__ static void
FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_29_block_kernel(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    int thread_id, int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(32, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = add(input2[tid], input3[tid]);
  float temp2 = add(temp1, temp0);
  output0[tid] = temp2;
}
// Node name:	Add_2319
// Description:	Add
// Input:
//	- name: Convolution_1008_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2864_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1016_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ static void
Add_float_float_float_cuda_Add_2319_block_kernel(float *input0, float *input1,
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_71(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1003_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 255) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1004_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 128 + 0, NULL);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 287) {
    FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_29_block_kernel(
        input5, input4, input7, input6, output2, threadIdx.x,
        blockIdx.x - 256 + 0, NULL);
  } else if ((int)blockIdx.x >= 288 && (int)blockIdx.x <= 319) {
    Add_float_float_float_cuda_Add_2319_block_kernel(
        input8, input9, output3, threadIdx.x, blockIdx.x - 288 + 0, NULL);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_71_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_71<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, input8, input9,
                                    output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_710_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2158_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3016_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_706_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_712_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2164_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3020_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_711_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2161_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3018_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Slice_708_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_731_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_709_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_733_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_732_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_713_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3015<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_710_0, Constant_2158_0,
// Constant_3016_0, Relu_731_0); Add_float_float_float_cuda_Add_709<<<dim3(64,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_706_0, AvgPool_706_0, Add_709_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3019<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_712_0, Constant_2164_0,
// Constant_3020_0, Relu_733_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3017<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_711_0, Constant_2161_0,
// Constant_3018_0, Relu_732_0); Relu_float_float_cuda_Relu_713<<<dim3(64, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Slice_708_0, Relu_713_0); Deduped function map:
// <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3019 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3015
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3017 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3015

// Node name:	Matched_Pattern_3015
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_710_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2158_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3016_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_731_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ static void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3015_block_kernel(
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
// Node name:	Add_709
// Description:	Add
// Input:
//	- name: AvgPool_706_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_706_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_709_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ static void
Add_float_float_float_cuda_Add_709_block_kernel(float *input0, float *input1,
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
// Node name:	Relu_713
// Description:	Relu
// Input:
//	- name: Slice_708_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_713_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ static void
Relu_float_float_cuda_Relu_713_block_kernel(float *input0, float *output0,
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
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Add_Matched_Pattern_Matched_Pattern_Relu_30(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {
  __shared__ char shared_buffer[3072];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3015_block_kernel(
        input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Add_float_float_float_cuda_Add_709_block_kernel(
        input3, input3, output1, threadIdx.x, blockIdx.x - 64 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3015_block_kernel(
        input4, input5, input6, output2, threadIdx.x, blockIdx.x - 128 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3015_block_kernel(
        input7, input8, input9, output3, threadIdx.x, blockIdx.x - 192 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319) {
    Relu_float_float_cuda_Relu_713_block_kernel(
        input10, output4, threadIdx.x, blockIdx.x - 256 + 0, shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Add_Matched_Pattern_Matched_Pattern_Relu_30_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Add_Matched_Pattern_Matched_Pattern_Relu_30<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1412_0	type: float	shape: Shape{1, 512, 8, 8}
//	- name: Constant_2542_0	type: float	shape: Shape{128, 512, 1, 1}
//	- name: Constant_2545_0	type: float	shape: Shape{128, 512, 1, 1}
// Output:
//	- name: Convolution_1414_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Convolution_1416_0	type: float	shape: Shape{1, 128, 8,
//8}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1414<<<dim3(1, 4, 16), dim3(8,
// 2, 8), 0, 0>>>(Relu_1412_0, Constant_2542_0, Convolution_1414_0);
// Convolution_float_float_float_cuda_Convolution_1416<<<dim3(1, 4, 16), dim3(8,
// 2, 8), 0, 0>>>(Relu_1412_0, Constant_2545_0, Convolution_1416_0); Deduped
// function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1416 :
// Convolution_float_float_float_cuda_Convolution_1414

// Node name:	Convolution_1414
// Description:	Convolution
// Input:
//	- name: Relu_1412_0	type: float	shape: Shape{1, 512, 8, 8}
//	- name: Constant_2542_0	type: float	shape: Shape{128, 512, 1, 1}
// Output:
//	- name: Convolution_1414_0	type: float	shape: Shape{1, 128, 8,
//8}
__device__ __forceinline__ static void
Convolution_float_float_float_cuda_Convolution_1414_block_kernel(
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
#pragma unroll
      for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        __syncthreads();
#pragma unroll
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0;
             ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2;
             ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
          pad_temp_shared[(
              (((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) +
               (((int)threadIdx.x) * 2)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] =
              input0[((((((rc_outer * 1024) + (((int)threadIdx.z) * 128)) +
                         (((int)threadIdx.y) * 64)) +
                        (((int)blockIdx.y) * 16)) +
                       (((int)threadIdx.x) * 2)) +
                      ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)];
        }
        input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) +
                       ((int)threadIdx.x))] =
            input1[(
                ((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 512)) +
                  (rc_outer * 16)) +
                 (((int)threadIdx.y) * 8)) +
                ((int)threadIdx.x))];
        __syncthreads();
#pragma unroll
        for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
          compute_local[0] =
              (compute_local[0] +
               (pad_temp_shared[(((rc_inner * 16) + (((int)threadIdx.y) * 8)) +
                                 ((int)threadIdx.x))] *
                input1_shared[((((int)threadIdx.z) * 16) + rc_inner)]));
        }
      }
      compute[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 64)) +
                 (((int)blockIdx.y) * 16)) +
                (((int)threadIdx.y) * 8)) +
               ((int)threadIdx.x))] = compute_local[0];
    }
  }
}
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_130(
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  __shared__ char shared_buffer[1536];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_1414_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_1414_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 64 + 0,
        shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_130_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_130<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2785_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1138_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: BatchNormInference_1076_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2786_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1140_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Slice_1094_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1147_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_1148_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_38<<<dim3(32, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1138_0, Constant_2785_0,
// BatchNormInference_1076_0, Add_1147_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_39<<<dim3(32, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1140_0, Constant_2786_0, Slice_1094_0,
// Add_1148_0); Deduped function map: <src_function_name :
// deduped_function_name> FusedKernel_float_float_float_float_cuda_Add_Add_39 :
// FusedKernel_float_float_float_float_cuda_Add_Add_38

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1138_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2785_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1076_0	type: float	shape: Shape{1,
//64, 16, 16}
// Output:
//	- name: Add_1147_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2397<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1138_0, Constant_2785_0, BatchNormInference_1144_0);
// Add_float_float_float_cuda_Add_1147<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1144_0, BatchNormInference_1076_0, Add_1147_0);
__device__ __forceinline__ static void
FusedKernel_float_float_float_float_cuda_Add_Add_38_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_91(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    FusedKernel_float_float_float_float_cuda_Add_Add_38_block_kernel(
        input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    FusedKernel_float_float_float_float_cuda_Add_Add_38_block_kernel(
        input4, input3, input5, output1, threadIdx.x, blockIdx.x - 32 + 0,
        NULL);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_91_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_91<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, output0, output1);
}
