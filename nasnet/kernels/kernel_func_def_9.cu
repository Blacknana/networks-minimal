// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_2926
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2926_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2926(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2926_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2926_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_356
// Description:	Constant
// Input:
// Output:
//	- name: Constant_356_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_356(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_356_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_356_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[3200];
  bin_file.read(tmp_mem, 3200);
  cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_129
// Description:	Constant
// Input:
// Output:
//	- name: Constant_129_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_129(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_129_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_129_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2383
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2383_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2383(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2383_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2383_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2545
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2545_0	type: float	shape: Shape{128, 512, 1, 1}
void Constant_float_cuda_Constant_2545(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2545_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2545_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[262144];
  bin_file.read(tmp_mem, 262144);
  cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2635
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2635_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2635(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2635_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2635_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3166
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3166_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3166(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3166_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3166_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2401
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2401_0	type: float	shape: Shape{64, 384, 1, 1}
void Constant_float_cuda_Constant_2401(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2401_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2401_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[98304];
  bin_file.read(tmp_mem, 98304);
  cudaMemcpyAsync(output0, tmp_mem, 98304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2801
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2801_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2801(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2801_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2801_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_364
// Description:	Constant
// Input:
// Output:
//	- name: Constant_364_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_364(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_364_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_364_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[3200];
  bin_file.read(tmp_mem, 3200);
  cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2227
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2227_0	type: float	shape: Shape{32, 192, 1, 1}
void Constant_float_cuda_Constant_2227(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2227_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2227_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[24576];
  bin_file.read(tmp_mem, 24576);
  cudaMemcpyAsync(output0, tmp_mem, 24576, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2260
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2260_0	type: float	shape: Shape{64, 192, 1, 1}
void Constant_float_cuda_Constant_2260(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2260_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2260_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[49152];
  bin_file.read(tmp_mem, 49152);
  cudaMemcpyAsync(output0, tmp_mem, 49152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_957_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2873_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: MaxPool_898_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_959_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2292<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_957_0, Constant_2873_0, BatchNormInference_958_0);
// Add_float_float_float_cuda_Add_959<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_958_0, MaxPool_898_0, Add_959_0);
extern "C" __launch_bounds__(512) __global__
    void FusedKernel_float_float_float_float_cuda_Add_Add_27(float *input0,
                                                             float *input1,
                                                             float *input2,
                                                             float *output0) {
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = add(temp0, input2[tid]);
  output0[tid] = temp1;
}
extern void FusedKernel_float_float_float_float_cuda_Add_Add_27_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0) {
  FusedKernel_float_float_float_float_cuda_Add_Add_27<<<grids, blocks, mem,
                                                        stream>>>(
      input0, input1, input2, output0);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2906_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1193_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2907_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1195_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Convolution_1197_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2908_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1190_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_351_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Relu_1191_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_175_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: Add_1207_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1202_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: DepthwiseConv2dNative_1198_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: DepthwiseConv2dNative_1199_0	type: float	shape: Shape{1,
//64, 16, 16}
// Fused functions:
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_41<<<dim3(32, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1193_0, Constant_2906_0,
// Convolution_1195_0, Constant_2907_0, Add_1207_0);
// Add_float_float_float_cuda_Add_2430<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1197_0, Constant_2908_0, BatchNormInference_1202_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1198<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1190_0, Constant_351_0,
// DepthwiseConv2dNative_1198_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1199<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1191_0, Constant_175_0,
// DepthwiseConv2dNative_1199_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1193_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2906_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1195_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2907_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1207_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2424<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1193_0, Constant_2906_0, BatchNormInference_1200_0);
// Add_float_float_float_cuda_Add_2427<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1195_0, Constant_2907_0, BatchNormInference_1201_0);
// Add_float_float_float_cuda_Add_1207<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1201_0, BatchNormInference_1200_0, Add_1207_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_41_block_kernel(
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
// Node name:	Add_2430
// Description:	Add
// Input:
//	- name: Convolution_1197_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2908_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1202_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2430_block_kernel(float *input0, float *input1,
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
// Node name:	DepthwiseConv2dNative_1198
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1190_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_351_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1198_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1198_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1199
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1191_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_175_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1199_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1199_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_98(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_41_block_kernel(
        input1, input0, input3, input2, output0, threadIdx.x, blockIdx.x - 0,
        NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    Add_float_float_float_cuda_Add_2430_block_kernel(
        input4, input5, output1, threadIdx.x, blockIdx.x - 32, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 191) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1198_block_kernel(
        input6, input7, output2, threadIdx.x, blockIdx.x - 64, NULL);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 319) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1199_block_kernel(
        input8, input9, output3, threadIdx.x, blockIdx.x - 192, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_98_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_98<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, input8, input9,
                                    output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: AvgPool_646_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_578_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Relu_647_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_364_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Constant_204_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_672_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_410_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_670_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_245_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Relu_671_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_192_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: Add_652_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_653_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_654_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_677_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_675_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_676_0	type: float	shape: Shape{1,
//32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_652<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(AvgPool_646_0, BatchNormInference_578_0, Add_652_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_653<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_647_0, Constant_364_0,
// DepthwiseConv2dNative_653_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_647_0, Constant_204_0,
// DepthwiseConv2dNative_654_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_677<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_672_0, Constant_410_0,
// DepthwiseConv2dNative_677_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_675<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_670_0, Constant_245_0,
// DepthwiseConv2dNative_675_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_676<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_671_0, Constant_192_0,
// DepthwiseConv2dNative_676_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_677 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_675 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_653
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_676 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654

// Node name:	Add_652
// Description:	Add
// Input:
//	- name: AvgPool_646_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_578_0	type: float	shape: Shape{1,
//32, 32, 32}
// Output:
//	- name: Add_652_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void Add_float_float_float_cuda_Add_652_block_kernel(
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
// Node name:	DepthwiseConv2dNative_653
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_647_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_364_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_653_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_653_block_kernel(
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
// Node name:	DepthwiseConv2dNative_654
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_647_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_204_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_654_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654_block_kernel(
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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_22(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Add_float_float_float_cuda_Add_652_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 319) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_653_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64, NULL);
  } else if ((int)blockIdx.x >= 320 && (int)blockIdx.x <= 575) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654_block_kernel(
        input2, input4, output2, threadIdx.x, blockIdx.x - 320, NULL);
  } else if ((int)blockIdx.x >= 576 && (int)blockIdx.x <= 831) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654_block_kernel(
        input5, input6, output3, threadIdx.x, blockIdx.x - 576, NULL);
  } else if ((int)blockIdx.x >= 832 && (int)blockIdx.x <= 1087) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_653_block_kernel(
        input7, input8, output4, threadIdx.x, blockIdx.x - 832, NULL);
  } else if ((int)blockIdx.x >= 1088 && (int)blockIdx.x <= 1343) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654_block_kernel(
        input9, input10, output5, threadIdx.x, blockIdx.x - 1088, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_22_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_22<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4, output5);
}
