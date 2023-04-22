// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_2620
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2620_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2620(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2620_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2620_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_62
// Description:	Constant
// Input:
// Output:
//	- name: Constant_62_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_62(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_62_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_62_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2548
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2548_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2548(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2548_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2548_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3118
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3118_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3118(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3118_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3118_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2659
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2659_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2659(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2659_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2659_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_1935
// Description:	Constant
// Input:
// Output:
//	- name: Constant_1935_0	type: float	shape: Shape{64, 384, 1, 1}
void Constant_float_cuda_Constant_1935(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_1935_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_1935_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[98304];
  bin_file.read(tmp_mem, 98304);
  cudaMemcpyAsync(output0, tmp_mem, 98304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2422
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2422_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2422(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2422_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2422_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2410
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2410_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2410(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2410_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2410_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2248
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2248_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2248(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2248_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2248_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2831
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2831_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2831(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2831_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2831_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_1644_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2958_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1646_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2849_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2848_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1648_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Relu_1641_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_159_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Relu_1642_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_13_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: BatchNormInference_1651_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Add_1658_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1649_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1650_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2673<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1644_0, Constant_2958_0, BatchNormInference_1651_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_66<<<dim3(16, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1646_0, Constant_2849_0,
// Convolution_1648_0, Constant_2848_0, Add_1658_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1649<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1641_0, Constant_159_0,
// DepthwiseConv2dNative_1649_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1650<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1642_0, Constant_13_0,
// DepthwiseConv2dNative_1650_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	Add_2673
// Description:	Add
// Input:
//	- name: Convolution_1644_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2958_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1651_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2673_block_kernel(float *input0, float *input1,
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
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1646_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2849_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1648_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2848_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1658_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2676<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1646_0, Constant_2849_0, BatchNormInference_1652_0);
// Add_float_float_float_cuda_Add_2679<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1648_0, Constant_2848_0, BatchNormInference_1653_0);
// Add_float_float_float_cuda_Add_1658<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1653_0, BatchNormInference_1652_0, Add_1658_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_66_block_kernel(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    int thread_id, int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(16, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = add(input2[tid], input3[tid]);
  float temp2 = add(temp1, temp0);
  output0[tid] = temp2;
}
// Node name:	DepthwiseConv2dNative_1649
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1641_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_159_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1649_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1649_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1650
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1642_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_13_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1650_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1650_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_fused_kernel_DepthwiseConv2dNative_DepthwiseConv2dNative_163(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    Add_float_float_float_cuda_Add_2673_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31) {
    FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_66_block_kernel(
        input2, input3, input5, input4, output1, threadIdx.x, blockIdx.x - 16,
        NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 95) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1649_block_kernel(
        input6, input7, output2, threadIdx.x, blockIdx.x - 32, NULL);
  } else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 159) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1650_block_kernel(
        input8, input9, output3, threadIdx.x, blockIdx.x - 96, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_fused_kernel_DepthwiseConv2dNative_DepthwiseConv2dNative_163_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_fused_kernel_DepthwiseConv2dNative_DepthwiseConv2dNative_163<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, input8, input9,
                                    output0, output1, output2, output3);
}
// Node name:	Constant_2880
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2880_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2880(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2880_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2880_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	 BlockFusion
// Input:
//	- name: Relu_762_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2188_0	type: float	shape: Shape{32, 192, 1, 1}
//	- name: Constant_2191_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_764_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Convolution_766_0	type: float	shape: Shape{1, 32, 32,
// 32}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_764<<<dim3(1, 32, 2), dim3(32,
// 1, 8), 0, 0>>>(Relu_762_0, Constant_2188_0, Convolution_764_0);
// Convolution_float_float_float_cuda_Convolution_766<<<dim3(1, 32, 2), dim3(32,
// 1, 8), 0, 0>>>(Relu_762_0, Constant_2191_0, Convolution_766_0); Deduped
// function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_766 :
// Convolution_float_float_float_cuda_Convolution_764

// Node name:	Convolution_764
// Description:	Convolution
// Input:
//	- name: Relu_762_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2188_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_764_0	type: float	shape: Shape{1, 32, 32,
// 32}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_764_block_kernel(
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
__device__ __forceinline__ void
fuse2_Convolution_float_float_float_cuda_Convolution_764_block_kernel(
    float *input0, float *input1, float *input2, float *output0, float *output1,
    int thread_id, int block_id, char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }

  //     int offset = block_id * 3072 + thread_id * 12;
  // #pragma unroll
  //     for (unsigned i = 0; i < 12; ++i) {
  //         input0[offset + i] = fmaxf(0, input0[offset + i]);
  //     }
  //     __syncthreads();
  const dim3 blockDim(32, 1, 8);
  const dim3 gridDim(1, 32, 2);
  const dim3 threadIdx(thread_id % 32, 0, thread_id / 32);
  const dim3 blockIdx(block_id % 1, block_id / 1 % 32, block_id / 32);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 6144);
  float *input1_shared1 = (float *)(shared_buffer + 9216);
  {
    float *compute = output0;
    {
      float compute_local[2];
      float compute_local1[2];

      compute_local[0] = 0.000000e+00f;
      compute_local[1] = 0.000000e+00f;
      compute_local1[0] = 0.000000e+00f;
      compute_local1[1] = 0.000000e+00f;
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
      input1_shared1[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] =
          input2[((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                   ((((int)threadIdx.x) >> 4) * 192)) +
                  ((((int)threadIdx.x) & 15) * 3))];
      input1_shared1[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                   ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) +
                  (((((int)threadIdx.x) * 3) + 1) % 48))];
      input1_shared1[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
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
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 96)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[((int)threadIdx.x)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 1)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 49)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 2)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 50)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 3)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 51)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 4)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 52)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 5)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 53)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 6)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 54)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 7)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 55)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 8)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 56)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 9)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 57)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 10)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 58)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 11)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 59)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 12)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 60)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 13)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 61)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 14)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 62)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 15)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 63)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 16)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 64)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 17)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 65)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 18)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 66)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 19)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 67)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 20)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 68)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 21)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 69)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 22)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 70)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 23)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 71)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 24)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 72)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 25)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 73)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 26)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 74)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 27)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 75)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 28)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 76)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 29)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 77)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 30)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 78)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 31)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 79)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 32)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 80)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 33)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 81)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 34)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 82)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 35)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 83)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 36)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 84)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 37)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 85)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 38)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 86)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 39)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 87)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 40)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 88)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 41)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 89)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 42)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 90)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 43)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 91)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 44)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 92)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 45)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 93)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 46)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 94)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 47)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 95)]));

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
      input1_shared1[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] =
          input2[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((int)threadIdx.x) >> 4) * 192)) +
                   ((((int)threadIdx.x) & 15) * 3)) +
                  48)];
      input1_shared1[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 1) % 48)) +
                  48)];
      input1_shared1[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
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
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 96)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[((int)threadIdx.x)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 1)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 49)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 2)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 50)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 3)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 51)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 4)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 52)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 5)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 53)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 6)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 54)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 7)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 55)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 8)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 56)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 9)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 57)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 10)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 58)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 11)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 59)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 12)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 60)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 13)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 61)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 14)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 62)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 15)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 63)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 16)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 64)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 17)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 65)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 18)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 66)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 19)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 67)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 20)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 68)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 21)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 69)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 22)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 70)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 23)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 71)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 24)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 72)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 25)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 73)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 26)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 74)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 27)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 75)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 28)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 76)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 29)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 77)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 30)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 78)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 31)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 79)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 32)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 80)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 33)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 81)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 34)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 82)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 35)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 83)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 36)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 84)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 37)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 85)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 38)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 86)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 39)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 87)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 40)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 88)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 41)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 89)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 42)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 90)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 43)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 91)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 44)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 92)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 45)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 93)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 46)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 94)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 47)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 95)]));

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
      input1_shared1[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] =
          input2[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((int)threadIdx.x) >> 4) * 192)) +
                   ((((int)threadIdx.x) & 15) * 3)) +
                  96)];
      input1_shared1[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 1) % 48)) +
                  96)];
      input1_shared1[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
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
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 96)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[((int)threadIdx.x)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 1)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 49)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 2)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 50)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 3)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 51)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 4)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 52)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 5)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 53)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 6)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 54)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 7)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 55)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 8)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 56)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 9)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 57)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 10)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 58)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 11)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 59)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 12)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 60)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 13)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 61)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 14)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 62)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 15)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 63)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 16)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 64)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 17)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 65)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 18)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 66)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 19)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 67)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 20)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 68)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 21)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 69)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 22)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 70)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 23)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 71)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 24)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 72)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 25)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 73)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 26)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 74)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 27)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 75)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 28)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 76)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 29)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 77)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 30)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 78)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 31)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 79)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 32)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 80)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 33)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 81)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 34)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 82)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 35)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 83)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 36)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 84)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 37)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 85)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 38)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 86)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 39)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 87)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 40)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 88)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 41)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 89)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 42)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 90)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 43)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 91)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 44)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 92)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 45)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 93)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 46)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 94)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 47)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 95)]));

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
      input1_shared1[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] =
          input2[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((int)threadIdx.x) >> 4) * 192)) +
                   ((((int)threadIdx.x) & 15) * 3)) +
                  144)];
      input1_shared1[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                      1)] =
          input2[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
                    ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) +
                   (((((int)threadIdx.x) * 3) + 1) % 48)) +
                  144)];
      input1_shared1[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) +
                      2)] =
          input2[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) +
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
      compute_local1[0] =
          (compute_local1[0] + (pad_temp_shared[((int)threadIdx.x)] *
                                input1_shared1[(((int)threadIdx.z) * 96)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[((int)threadIdx.x)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 48)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 1)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 32)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 49)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 2)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 64)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 50)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 3)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 96)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 51)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 4)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 128)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 52)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 5)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 160)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 53)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 6)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 192)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 54)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 7)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 224)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 55)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 8)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 256)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 56)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 9)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 288)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 57)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 10)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 320)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 58)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 11)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 352)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 59)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 12)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 384)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 60)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 13)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 416)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 61)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 14)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 448)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 62)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 15)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 480)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 63)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 16)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 512)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 64)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 17)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 544)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 65)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 18)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 576)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 66)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 19)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 608)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 67)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 20)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 640)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 68)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 21)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 672)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 69)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 22)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 704)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 70)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 23)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 736)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 71)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 24)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 768)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 72)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 25)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 800)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 73)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 26)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 832)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 74)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 27)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 864)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 75)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 28)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 896)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 76)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 29)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 928)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 77)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 30)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 960)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 78)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 31)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 992)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 79)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 32)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1024)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 80)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 33)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1056)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 81)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 34)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1088)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 82)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 35)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1120)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 83)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 36)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1152)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 84)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 37)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1184)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 85)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 38)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1216)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 86)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 39)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1248)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 87)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 40)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1280)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 88)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 41)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1312)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 89)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 42)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1344)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 90)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 43)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1376)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 91)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 44)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1408)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 92)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 45)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1440)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 93)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 46)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1472)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 94)]));
      compute_local1[0] = (compute_local1[0] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 47)]));
      compute_local1[1] = (compute_local1[1] +
                           (pad_temp_shared[(((int)threadIdx.x) + 1504)] *
                            input1_shared1[((((int)threadIdx.z) * 96) + 95)]));

      compute[((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) +
                (((int)blockIdx.y) * 32)) +
               ((int)threadIdx.x))] = compute_local[0];
      compute[(((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) +
                 (((int)blockIdx.y) * 32)) +
                ((int)threadIdx.x)) +
               1024)] = compute_local[1];
      output1[((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) +
                (((int)blockIdx.y) * 32)) +
               ((int)threadIdx.x))] = compute_local1[0];
      output1[(((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) +
                 (((int)blockIdx.y) * 32)) +
                ((int)threadIdx.x)) +
               1024)] = compute_local1[1];
    }
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_36(
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {

  __shared__ char shared_buffer[12288];
  //__shared__ char shared_buffer[9216];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    fuse2_Convolution_float_float_float_cuda_Convolution_764_block_kernel(
        input0, input1, input2, output0, output1, threadIdx.x, blockIdx.x - 0,
        shared_buffer);
    // Convolution_float_float_float_cuda_Convolution_764_block_kernel(input0,
    // input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  }
  // else if((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
  // {
  //     Convolution_float_float_float_cuda_Convolution_764_block_kernel(input0,
  //     input2, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  // }
}

extern void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_36_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_36<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1701_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_281_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Relu_1702_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_345_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Convolution_1709_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2851_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2800_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1711_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Convolution_1713_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2967_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: DepthwiseConv2dNative_1706_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1707_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: BatchNormInference_1718_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Add_1723_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1706<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1701_0, Constant_281_0,
// DepthwiseConv2dNative_1706_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1707<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1702_0, Constant_345_0,
// DepthwiseConv2dNative_1707_0); Add_float_float_float_cuda_Add_2709<<<dim3(16,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1709_0, Constant_2851_0,
// BatchNormInference_1718_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_70<<<dim3(16, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1711_0, Constant_2800_0,
// Convolution_1713_0, Constant_2967_0, Add_1723_0); Deduped function map:
// <src_function_name : deduped_function_name>

// Node name:	DepthwiseConv2dNative_1706
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1701_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_281_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1706_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1706_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1707
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1702_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_345_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1707_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1707_block_kernel(
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
// Node name:	Add_2709
// Description:	Add
// Input:
//	- name: Convolution_1709_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2851_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1718_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2709_block_kernel(float *input0, float *input1,
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
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1711_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2800_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1713_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2967_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1723_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2712<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1711_0, Constant_2800_0, BatchNormInference_1719_0);
// Add_float_float_float_cuda_Add_2715<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1713_0, Constant_2967_0, BatchNormInference_1720_0);
// Add_float_float_float_cuda_Add_1723<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1720_0, BatchNormInference_1719_0, Add_1723_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_70_block_kernel(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    int thread_id, int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(16, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = add(input2[tid], input3[tid]);
  float temp2 = add(temp1, temp0);
  output0[tid] = temp2;
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_172(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    Add_float_float_float_cuda_Add_2709_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31) {
    FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_70_block_kernel(
        input7, input6, input8, input9, output3, threadIdx.x, blockIdx.x - 16,
        NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 95) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1706_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 32, NULL);
  } else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 159) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1707_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 96, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_172_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_172<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, input8, input9,
                                    output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1213_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2437_0	type: float	shape: Shape{64, 384, 1, 1}
//	- name: Constant_2440_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1215_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Convolution_1217_0	type: float	shape: Shape{1, 64, 16,
// 16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1215<<<dim3(1, 16, 4),
// dim3(16, 1, 16), 0, 0>>>(Relu_1213_0, Constant_2437_0, Convolution_1215_0);
// Convolution_float_float_float_cuda_Convolution_1217<<<dim3(1, 16, 4),
// dim3(16, 1, 16), 0, 0>>>(Relu_1213_0, Constant_2440_0, Convolution_1217_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1217 :
// Convolution_float_float_float_cuda_Convolution_1215

// Node name:	Convolution_1215
// Description:	Convolution
// Input:
//	- name: Relu_1213_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2437_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1215_0	type: float	shape: Shape{1, 64, 16,
// 16}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1215_block_kernel(
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
          relu(input0[((((((int)threadIdx.z) * 768) +
                         (((((int)threadIdx.x) * 3) / 16) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       ((((int)threadIdx.x) * 3) & 15))]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       (((((int)threadIdx.x) * 3) + 1) & 15))]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[((((((int)threadIdx.z) * 768) +
                         ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                        (((int)blockIdx.y) * 16)) +
                       (((((int)threadIdx.x) * 3) + 2) & 15))]);
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
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       12288)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       12288)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       12288)]);
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
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       24576)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       24576)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       24576)]);
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
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       36864)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       36864)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       36864)]);
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
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       49152)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       49152)]);
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
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       61440)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       61440)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       61440)]);
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
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       73728)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       73728)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       73728)]);
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
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          (((((int)threadIdx.x) * 3) / 16) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        ((((int)threadIdx.x) * 3) & 15)) +
                       86016)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       1)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 1) & 15)) +
                       86016)]);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) +
                       2)] =
          relu(input0[(((((((int)threadIdx.z) * 768) +
                          ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) +
                         (((int)blockIdx.y) * 16)) +
                        (((((int)threadIdx.x) * 3) + 2) & 15)) +
                       86016)]);
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
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_101(
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {

  __shared__ char shared_buffer[6144];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_1215_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_1215_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_101_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_101<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
