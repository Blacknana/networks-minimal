// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	Constant_2963
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2963_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2963(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2963_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2963_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2029
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2029_0	type: float	shape: Shape{32, 96, 1, 1}
void Constant_float_cuda_Constant_2029(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2029_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2029_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12288];
  bin_file.read(tmp_mem, 12288);
  cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_189
// Description:	Constant
// Input:
// Output:
//	- name: Constant_189_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_189(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_189_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_189_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2443
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2443_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2443(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2443_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2443_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3088
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3088_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3088(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3088_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3088_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2452
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2452_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2452(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2452_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2452_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_443
// Description:	Constant
// Input:
// Output:
//	- name: Constant_443_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_443(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_443_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_443_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_230
// Description:	Constant
// Input:
// Output:
//	- name: Constant_230_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_230(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_230_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_230_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_143
// Description:	Constant
// Input:
// Output:
//	- name: Constant_143_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_143(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_143_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_143_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2899
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2899_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2899(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2899_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2899_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2827
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2827_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2827(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2827_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2827_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3026
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3026_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_3026(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3026_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3026_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2808_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_535_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2809_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_537_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: AvgPool_509_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_490_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Relu_510_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_302_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Constant_217_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Relu_541_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_286_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Relu_540_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_61_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_539_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_375_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Convolution_526_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2806_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_551_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_520_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_521_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_522_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_548_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_547_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_546_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: BatchNormInference_538_0	type: float	shape: Shape{1,
//32, 32, 32}
// Fused functions:
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_2<<<dim3(64, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_535_0, Constant_2808_0,
// Convolution_537_0, Constant_2809_0, Add_551_0);
// Add_float_float_float_cuda_Add_520<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(AvgPool_509_0, BatchNormInference_490_0, Add_520_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_521<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_510_0, Constant_302_0,
// DepthwiseConv2dNative_521_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_522<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_510_0, Constant_217_0,
// DepthwiseConv2dNative_522_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_548<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_541_0, Constant_286_0,
// DepthwiseConv2dNative_548_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_547<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_540_0, Constant_61_0,
// DepthwiseConv2dNative_547_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_546<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_539_0, Constant_375_0,
// DepthwiseConv2dNative_546_0); Add_float_float_float_cuda_Add_2052<<<dim3(64,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_526_0, Constant_2806_0,
// BatchNormInference_538_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_548 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_522
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_547 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_521
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_546 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_521
// Add_float_float_float_cuda_Add_2052 : Add_float_float_float_cuda_Add_520

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_535_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2808_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_537_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2809_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_551_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2061<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_535_0, Constant_2808_0, BatchNormInference_544_0);
// Add_float_float_float_cuda_Add_2064<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_537_0, Constant_2809_0, BatchNormInference_545_0);
// Add_float_float_float_cuda_Add_551<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_545_0, BatchNormInference_544_0, Add_551_0);
__device__ __forceinline__ static void
FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_2_block_kernel(
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
  float temp2 = add(temp1, temp0);
  output0[tid] = temp2;
}
// Node name:	Add_520
// Description:	Add
// Input:
//	- name: AvgPool_509_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_490_0	type: float	shape: Shape{1,
//32, 32, 32}
// Output:
//	- name: Add_520_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ static void
Add_float_float_float_cuda_Add_520_block_kernel(float *input0, float *input1,
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
// Node name:	DepthwiseConv2dNative_521
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_510_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_302_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_521_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ static void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_521_block_kernel(
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
// Node name:	DepthwiseConv2dNative_522
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_510_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_217_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_522_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ static void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_522_block_kernel(
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
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_4(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *input12, float *input13,
    float *input14, float *input15, float *input16, float *output0,
    float *output1, float *output2, float *output3, float *output4,
    float *output5, float *output6, float *output7) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_2_block_kernel(
        input1, input0, input3, input2, output0, threadIdx.x,
        blockIdx.x - 0 + 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Add_float_float_float_cuda_Add_520_block_kernel(
        input4, input5, output1, threadIdx.x, blockIdx.x - 64 + 0, NULL);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 383) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_521_block_kernel(
        input6, input7, output2, threadIdx.x, blockIdx.x - 128 + 0, NULL);
  } else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 639) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_522_block_kernel(
        input6, input8, output3, threadIdx.x, blockIdx.x - 384 + 0, NULL);
  } else if ((int)blockIdx.x >= 640 && (int)blockIdx.x <= 895) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_522_block_kernel(
        input9, input10, output4, threadIdx.x, blockIdx.x - 640 + 0, NULL);
  } else if ((int)blockIdx.x >= 896 && (int)blockIdx.x <= 1151) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_521_block_kernel(
        input11, input12, output5, threadIdx.x, blockIdx.x - 896 + 0, NULL);
  } else if ((int)blockIdx.x >= 1152 && (int)blockIdx.x <= 1407) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_521_block_kernel(
        input13, input14, output6, threadIdx.x, blockIdx.x - 1152 + 0, NULL);
  } else if ((int)blockIdx.x >= 1408 && (int)blockIdx.x <= 1471) {
    Add_float_float_float_cuda_Add_520_block_kernel(
        input15, input16, output7, threadIdx.x, blockIdx.x - 1408 + 0, NULL);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_4_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *input11, float *input12, float *input13,
    float *input14, float *input15, float *input16, float *output0,
    float *output1, float *output2, float *output3, float *output4,
    float *output5, float *output6, float *output7) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_4<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, input11, input12, input13, input14, input15, input16,
      output0, output1, output2, output3, output4, output5, output6, output7);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_963_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2749_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2874_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_965_0	type: float	shape: Shape{1, 64, 16,
//16}
// Output:
//	- name: BatchNormInference_966_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: BatchNormInference_967_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Relu_969_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2295<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_963_0, Constant_2749_0, BatchNormInference_966_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_28<<<dim3(32, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_965_0, Constant_2874_0, Relu_969_0,
// BatchNormInference_967_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	Add_2295
// Description:	Add
// Input:
//	- name: Convolution_963_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2749_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_966_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ static void
Add_float_float_float_cuda_Add_2295_block_kernel(float *input0, float *input1,
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
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_965_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2874_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_969_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_967_0	type: float	shape: Shape{1,
//64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2298<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_965_0, Constant_2874_0, BatchNormInference_967_0);
// Relu_float_float_cuda_Relu_969<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_967_0, Relu_969_0);
__device__ __forceinline__ static void
FusedKernel_float_float_float_float_cuda_Add_Relu_28_block_kernel(
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
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_66(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_2295_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    FusedKernel_float_float_float_float_cuda_Add_Relu_28_block_kernel(
        input3, input2, output2, output1, threadIdx.x, blockIdx.x - 32 + 0,
        NULL);
  }
}
extern void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_66_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {
  BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_66<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1, output2);
}
// Node name:	Add_1762
// Description:	Add
// Input:
//	- name: Dot_1760_0	type: float	shape: Shape{1, 10}
//	- name: Constant_2010_0	type: float	shape: Shape{1, 10}
// Output:
//	- name: Add_1762_0	type: float	shape: Shape{1, 10}
extern "C" __launch_bounds__(10) __global__
    void Add_float_float_float_cuda_Add_1762(float *input0, float *input1,
                                             float *output0) {
  output0[threadIdx.x] = add(input0[threadIdx.x], input1[threadIdx.x]);
}
extern void Add_float_float_float_cuda_Add_1762_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *output0) {
  Add_float_float_float_cuda_Add_1762<<<grids, blocks, mem, stream>>>(
      input0, input1, output0);
}
