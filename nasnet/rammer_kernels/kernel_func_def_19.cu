// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	Constant_3178
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3178_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3178(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3178_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3178_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2892
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2892_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2892(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2892_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2892_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_437
// Description:	Constant
// Input:
// Output:
//	- name: Constant_437_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_437(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_437_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_437_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_186
// Description:	Constant
// Input:
// Output:
//	- name: Constant_186_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_186(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_186_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_186_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3144
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3144_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3144(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3144_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3144_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2853
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2853_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2853(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2853_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2853_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_425
// Description:	Constant
// Input:
// Output:
//	- name: Constant_425_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_425(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_425_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_425_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2988
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2988_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2988(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2988_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2988_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2137
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2137_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2137(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2137_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2137_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2230
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2230_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2230(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2230_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2230_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_799_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_237_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_800_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_431_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Constant_2844_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_807_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Convolution_811_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2771_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_809_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2783_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: DepthwiseConv2dNative_804_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_805_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Add_821_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_817_0	type: float	shape: Shape{1,
//32, 32, 32}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_804<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_799_0, Constant_237_0,
// DepthwiseConv2dNative_804_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_805<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_800_0, Constant_431_0,
// DepthwiseConv2dNative_805_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_20<<<dim3(64, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_807_0, Constant_2844_0,
// Convolution_811_0, Constant_2771_0, Add_821_0);
// Add_float_float_float_cuda_Add_2214<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_809_0, Constant_2783_0, BatchNormInference_817_0); Deduped
// function map: <src_function_name : deduped_function_name>

// Node name:	DepthwiseConv2dNative_804
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_799_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_237_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_804_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ static void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_804_block_kernel(
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
// Node name:	DepthwiseConv2dNative_805
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_800_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_431_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_805_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ static void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_805_block_kernel(
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
//	- name: Convolution_807_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2844_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_811_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2771_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_821_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2211<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_807_0, Constant_2844_0, BatchNormInference_816_0);
// Add_float_float_float_cuda_Add_2217<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_811_0, Constant_2771_0, BatchNormInference_818_0);
// Add_float_float_float_cuda_Add_821<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_816_0, BatchNormInference_818_0, Add_821_0);
__device__ __forceinline__ static void
FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_20_block_kernel(
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
  float temp2 = add(temp0, temp1);
  output0[tid] = temp2;
}
// Node name:	Add_2214
// Description:	Add
// Input:
//	- name: Convolution_809_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2783_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_817_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ static void
Add_float_float_float_cuda_Add_2214_block_kernel(float *input0, float *input1,
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_42(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 255) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_804_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 511) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_805_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 256 + 0, NULL);
  } else if ((int)blockIdx.x >= 512 && (int)blockIdx.x <= 575) {
    FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_20_block_kernel(
        input5, input4, input6, input7, output2, threadIdx.x,
        blockIdx.x - 512 + 0, NULL);
  } else if ((int)blockIdx.x >= 576 && (int)blockIdx.x <= 639) {
    Add_float_float_float_cuda_Add_2214_block_kernel(
        input8, input9, output3, threadIdx.x, blockIdx.x - 576 + 0, NULL);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_42_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_42<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, input8, input9,
                                    output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_1321_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2924_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1317_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2923_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1319_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2922_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1296_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2488_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3116_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1297_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2491_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3118_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1326_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: BatchNormInference_1324_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: BatchNormInference_1325_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Relu_1314_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1315_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2502<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1321_0, Constant_2924_0, BatchNormInference_1326_0);
// Add_float_float_float_cuda_Add_2496<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1317_0, Constant_2923_0, BatchNormInference_1324_0);
// Add_float_float_float_cuda_Add_2499<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1319_0, Constant_2922_0, BatchNormInference_1325_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3115<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_1296_0,
// Constant_2488_0, Constant_3116_0, Relu_1314_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3117<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_1297_0,
// Constant_2491_0, Constant_3118_0, Relu_1315_0); Deduped function map:
// <src_function_name : deduped_function_name>
// Add_float_float_float_cuda_Add_2496 : Add_float_float_float_cuda_Add_2502
// Add_float_float_float_cuda_Add_2499 : Add_float_float_float_cuda_Add_2502
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3117 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3115

// Node name:	Add_2502
// Description:	Add
// Input:
//	- name: Convolution_1321_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2924_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1326_0	type: float	shape: Shape{1,
//128, 8, 8}
__device__ __forceinline__ static void
Add_float_float_float_cuda_Add_2502_block_kernel(float *input0, float *input1,
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
// Node name:	Matched_Pattern_3115
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1296_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2488_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3116_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1314_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ static void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3115_block_kernel(
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
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Add_Add_Matched_Pattern_Matched_Pattern_115(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {
  __shared__ char shared_buffer[2048];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    Add_float_float_float_cuda_Add_2502_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_2502_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 16 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 47) {
    Add_float_float_float_cuda_Add_2502_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 32 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 48 && (int)blockIdx.x <= 111) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3115_block_kernel(
        input6, input7, input8, output3, threadIdx.x, blockIdx.x - 48 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 112 && (int)blockIdx.x <= 175) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3115_block_kernel(
        input9, input10, input11, output4, threadIdx.x, blockIdx.x - 112 + 0,
        shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Add_Add_Matched_Pattern_Matched_Pattern_115_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *output0, float *output1,
    float *output2, float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Add_Add_Matched_Pattern_Matched_Pattern_115<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, input11, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: Add_903_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: AvgPool_937_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_946_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2278_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3056_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_934_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2281_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_935_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2284_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_936_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2287_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Add_945_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_954_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_940_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Convolution_942_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Convolution_944_0	type: float	shape: Shape{1, 64, 16,
//16}
// Fused functions:
// Add_float_float_float_cuda_Add_945<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Add_903_0, AvgPool_937_0, Add_945_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3055<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_946_0, Constant_2278_0,
// Constant_3056_0, Relu_954_0);
// Convolution_float_float_float_cuda_Convolution_940<<<dim3(1, 16, 4), dim3(8,
// 1, 16), 0, 0>>>(DepthwiseConv2dNative_934_0, Constant_2281_0,
// Convolution_940_0);
// Convolution_float_float_float_cuda_Convolution_942<<<dim3(1, 16, 4), dim3(8,
// 1, 16), 0, 0>>>(DepthwiseConv2dNative_935_0, Constant_2284_0,
// Convolution_942_0);
// Convolution_float_float_float_cuda_Convolution_944<<<dim3(1, 16, 4), dim3(8,
// 1, 16), 0, 0>>>(DepthwiseConv2dNative_936_0, Constant_2287_0,
// Convolution_944_0); Deduped function map: <src_function_name :
// deduped_function_name> Convolution_float_float_float_cuda_Convolution_942 :
// Convolution_float_float_float_cuda_Convolution_940
// Convolution_float_float_float_cuda_Convolution_944 :
// Convolution_float_float_float_cuda_Convolution_940

// Node name:	Add_945
// Description:	Add
// Input:
//	- name: Add_903_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: AvgPool_937_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_945_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ static void
Add_float_float_float_cuda_Add_945_block_kernel(float *input0, float *input1,
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
// Node name:	Matched_Pattern_3055
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_946_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2278_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3056_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_954_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ static void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3055_block_kernel(
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
// Node name:	Convolution_940
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_934_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Constant_2281_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_940_0	type: float	shape: Shape{1, 64, 16,
//16}
__device__ __forceinline__ static void
Convolution_float_float_float_cuda_Convolution_940_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
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
      float compute_local[2];

      compute_local[0] = 0.000000e+00f;
      compute_local[1] = 0.000000e+00f;
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
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
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
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
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
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
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
      compute_local[0] =
          (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[1] =
          (compute_local[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] *
                               input1_shared[(((int)threadIdx.z) * 16)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 1)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 2)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 48)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 49)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 3)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 4)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 80)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 81)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 5)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 6)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 7)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 128)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 129)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 8)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 145)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 9)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 160)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 161)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 10)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 176)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 177)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 11)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 192)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 193)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 12)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 208)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 209)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 13)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 14)]));
      compute_local[0] = (compute_local[0] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 240)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute_local[1] = (compute_local[1] +
                          (pad_temp_shared[((((int)threadIdx.x) * 2) + 241)] *
                           input1_shared[((((int)threadIdx.z) * 16) + 15)]));
      compute[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                (((int)blockIdx.y) * 16)) +
               (((int)threadIdx.x) * 2))] = compute_local[0];
      compute[(((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) +
                 (((int)blockIdx.y) * 16)) +
                (((int)threadIdx.x) * 2)) +
               1)] = compute_local[1];
    }
  }
}
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Matched_Pattern_Convolution_Convolution_Convolution_63(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {
  __shared__ char shared_buffer[2048];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_945_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 95) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3055_block_kernel(
        input2, input3, input4, output1, threadIdx.x, blockIdx.x - 32 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 159) {
    Convolution_float_float_float_cuda_Convolution_940_block_kernel(
        input5, input6, output2, threadIdx.x, blockIdx.x - 96 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 160 && (int)blockIdx.x <= 223) {
    Convolution_float_float_float_cuda_Convolution_940_block_kernel(
        input7, input8, output3, threadIdx.x, blockIdx.x - 160 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 224 && (int)blockIdx.x <= 287) {
    Convolution_float_float_float_cuda_Convolution_940_block_kernel(
        input9, input10, output4, threadIdx.x, blockIdx.x - 224 + 0,
        shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Matched_Pattern_Convolution_Convolution_Convolution_63_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Matched_Pattern_Convolution_Convolution_Convolution_63<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_678_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2146_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_679_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2149_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_687_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Convolution_689_0	type: float	shape: Shape{1, 32, 32,
//32}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_687<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_678_0, Constant_2146_0,
// Convolution_687_0);
// Convolution_float_float_float_cuda_Convolution_689<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_679_0, Constant_2149_0,
// Convolution_689_0); Deduped function map: <src_function_name :
// deduped_function_name> Convolution_float_float_float_cuda_Convolution_689 :
// Convolution_float_float_float_cuda_Convolution_687

// Node name:	Convolution_687
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_678_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2146_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_687_0	type: float	shape: Shape{1, 32, 32,
//32}
__device__ __forceinline__ static void
Convolution_float_float_float_cuda_Convolution_687_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_25(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  __shared__ char shared_buffer[3072];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_687_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_687_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64 + 0,
        shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_25_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_25<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Slice_769_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_775_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2194_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3026_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_777_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2200_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3030_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_770_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_776_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2197_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3028_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_772_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_796_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_798_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_774_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_797_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Relu_float_float_cuda_Relu_772<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Slice_769_0, Relu_772_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3025<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_775_0, Constant_2194_0,
// Constant_3026_0, Relu_796_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3029<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_777_0, Constant_2200_0,
// Constant_3030_0, Relu_798_0); Add_float_float_float_cuda_Add_774<<<dim3(64,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_770_0, AvgPool_770_0, Add_774_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3027<<<dim3(2,
// 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_776_0, Constant_2197_0,
// Constant_3028_0, Relu_797_0); Deduped function map: <src_function_name :
// deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3029 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3025
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3027 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3025

// Node name:	Relu_772
// Description:	Relu
// Input:
//	- name: Slice_769_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_772_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ static void
Relu_float_float_cuda_Relu_772_block_kernel(float *input0, float *output0,
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
// Node name:	Matched_Pattern_3025
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_775_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2194_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3026_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_796_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ static void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3025_block_kernel(
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
// Node name:	Add_774
// Description:	Add
// Input:
//	- name: AvgPool_770_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_770_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_774_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ static void
Add_float_float_float_cuda_Add_774_block_kernel(float *input0, float *input1,
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Matched_Pattern_Matched_Pattern_Add_Matched_Pattern_39(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {
  __shared__ char shared_buffer[3072];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Relu_float_float_cuda_Relu_772_block_kernel(
        input0, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3025_block_kernel(
        input1, input2, input3, output1, threadIdx.x, blockIdx.x - 64 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3025_block_kernel(
        input4, input5, input6, output2, threadIdx.x, blockIdx.x - 128 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    Add_float_float_float_cuda_Add_774_block_kernel(
        input7, input7, output3, threadIdx.x, blockIdx.x - 192 + 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3025_block_kernel(
        input8, input9, input10, output4, threadIdx.x, blockIdx.x - 256 + 0,
        shared_buffer);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Matched_Pattern_Matched_Pattern_Add_Matched_Pattern_39_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Matched_Pattern_Matched_Pattern_Add_Matched_Pattern_39<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4);
}
