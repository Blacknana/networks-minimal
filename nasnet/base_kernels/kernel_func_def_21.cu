// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	Constant_2320
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2320_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2320(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2320_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2320_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_375
// Description:	Constant
// Input:
// Output:
//	- name: Constant_375_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_375(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_375_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_375_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2992
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2992_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2992(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2992_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2992_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2407
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2407_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2407(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2407_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2407_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_14
// Description:	Constant
// Input:
// Output:
//	- name: Constant_14_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_14(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_14_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_14_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2413
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2413_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2413(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2413_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2413_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2774
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2774_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2774(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2774_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2774_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2314
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2314_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2314(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2314_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2314_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2964
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2964_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2964(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2964_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2964_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_416
// Description:	Constant
// Input:
// Output:
//	- name: Constant_416_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_416(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_416_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_416_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[2304];
  bin_file.read(tmp_mem, 2304);
  cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2906
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2906_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2906(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2906_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2906_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: BatchNormInference_467_0	type: float	shape: Shape{1,
// 96, 32, 32}
//	- name: Constant_2014_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Relu_468_0	type: float	shape: Shape{1, 96, 32, 32}
//	- name: Constant_434_0	type: float	shape: Shape{5, 5, 96, 1}
//	- name: Constant_58_0	type: float	shape: Shape{3, 3, 96, 1}
//	- name: Constant_2029_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Constant_2026_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Constant_355_0	type: float	shape: Shape{3, 3, 96, 1}
//	- name: Constant_2023_0	type: float	shape: Shape{32, 96, 1, 1}
// Output:
//	- name: Convolution_471_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: DepthwiseConv2dNative_480_0	type: float	shape: Shape{1,
// 96, 32, 32}
//	- name: DepthwiseConv2dNative_479_0	type: float	shape: Shape{1,
// 96, 32, 32}
//	- name: Convolution_476_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Convolution_474_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: DepthwiseConv2dNative_472_0	type: float	shape: Shape{1,
// 96, 32, 32}
//	- name: Convolution_478_0	type: float	shape: Shape{1, 32, 32,
// 32}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_471<<<dim3(1, 32, 2), dim3(16,
// 1, 8), 0, 0>>>(BatchNormInference_467_0, Constant_2014_0, Convolution_471_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_480<<<dim3(768,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_468_0, Constant_434_0,
// DepthwiseConv2dNative_480_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_479<<<dim3(768,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_468_0, Constant_58_0,
// DepthwiseConv2dNative_479_0);
// Convolution_float_float_float_cuda_Convolution_476<<<dim3(1, 32, 2), dim3(16,
// 1, 8), 0, 0>>>(Relu_468_0, Constant_2029_0, Convolution_476_0);
// Convolution_float_float_float_cuda_Convolution_474<<<dim3(1, 32, 2), dim3(16,
// 1, 8), 0, 0>>>(Relu_468_0, Constant_2026_0, Convolution_474_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_472<<<dim3(768,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_468_0, Constant_355_0,
// DepthwiseConv2dNative_472_0);
// Convolution_float_float_float_cuda_Convolution_478<<<dim3(1, 32, 2), dim3(16,
// 1, 8), 0, 0>>>(Relu_468_0, Constant_2023_0, Convolution_478_0); Deduped
// function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_476 :
// Convolution_float_float_float_cuda_Convolution_471
// Convolution_float_float_float_cuda_Convolution_474 :
// Convolution_float_float_float_cuda_Convolution_471
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_472 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_479
// Convolution_float_float_float_cuda_Convolution_478 :
// Convolution_float_float_float_cuda_Convolution_471

// Node name:	Convolution_471
// Description:	Convolution
// Input:
//	- name: BatchNormInference_467_0	type: float	shape: Shape{1,
// 96, 32, 32}
//	- name: Constant_2014_0	type: float	shape: Shape{32, 96, 1, 1}
// Output:
//	- name: Convolution_471_0	type: float	shape: Shape{1, 32, 32,
// 32}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_471_block_kernel(
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
// Node name:	DepthwiseConv2dNative_480
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_468_0	type: float	shape: Shape{1, 96, 32, 32}
//	- name: Constant_434_0	type: float	shape: Shape{5, 5, 96, 1}
// Output:
//	- name: DepthwiseConv2dNative_480_0	type: float	shape: Shape{1,
// 96, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_480_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(128, 1, 1);
  const dim3 gridDim(768, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);

  typedef float S;
  float *input = input0;
  float *filter = input1;
  float *output = output0;

  const int in_height = 32;
  const int in_width = 32;
  const int in_depth = 96;
  const int filter_height = 5;
  const int filter_width = 5;
  const int depth_multiplier = 1;
  const int stride = 1;
  const int pad_height = 2;
  const int pad_width = 2;
  const int out_height = 32;
  const int out_width = 32;
  const int out_depth = 96;
  const int num_outputs = 98304;

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
// Node name:	DepthwiseConv2dNative_479
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_468_0	type: float	shape: Shape{1, 96, 32, 32}
//	- name: Constant_58_0	type: float	shape: Shape{3, 3, 96, 1}
// Output:
//	- name: DepthwiseConv2dNative_479_0	type: float	shape: Shape{1,
// 96, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_479_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(128, 1, 1);
  const dim3 gridDim(768, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);

  typedef float S;
  float *input = input0;
  float *filter = input1;
  float *output = output0;

  const int in_height = 32;
  const int in_width = 32;
  const int in_depth = 96;
  const int filter_height = 3;
  const int filter_width = 3;
  const int depth_multiplier = 1;
  const int stride = 1;
  const int pad_height = 1;
  const int pad_width = 1;
  const int out_height = 32;
  const int out_width = 32;
  const int out_depth = 96;
  const int num_outputs = 98304;

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
__device__ __forceinline__ void
Fuse_2_DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_479_block_kernel(
    float *input0, float *input1, float *input2, float *output0, float *output1,
    int thread_id, int block_id, char *shared_buffer) {
  if (thread_id >= 128) {
    return;
  }
  const dim3 blockDim(128, 1, 1);
  const dim3 gridDim(768, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);

  typedef float S;
  // float *input = input0;
  // float *filter = input1;
  // float *output = output0;

  const int in_height = 32;
  const int in_width = 32;
  const int in_depth = 96;
  const int filter_height = 3;
  const int filter_width = 3;
  const int depth_multiplier = 1;
  const int stride = 1;
  const int pad_height = 1;
  const int pad_width = 1;
  const int out_height = 32;
  const int out_width = 32;
  const int out_depth = 96;
  const int num_outputs = 98304;

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

    S sum0 = static_cast<S>(0);
    S sum1 = static_cast<S>(0);
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
          sum0 += static_cast<S>(__ldg(input0 + input_offset)) *
                  static_cast<S>(__ldg(input1 + filter_offset));
          sum1 += static_cast<S>(__ldg(input0 + input_offset)) *
                  static_cast<S>(__ldg(input2 + filter_offset));
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
            sum0 += static_cast<S>(__ldg(input0 + input_offset)) *
                    static_cast<S>(__ldg(input1 + filter_offset));
            sum1 += static_cast<S>(__ldg(input0 + input_offset)) *
                    static_cast<S>(__ldg(input2 + filter_offset));
          }
        }
      }
    }

    output0[thread_id] = static_cast<S>(sum0);
    output1[thread_id] = static_cast<S>(sum1);
  }
}
__device__ __forceinline__ void
fused3_Convolution_float_float_float_cuda_Convolution_471_block_kernel1(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2, int thread_id, int block_id,
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
  float *input1_shared1 = (float *)(shared_buffer + 3072);
  float *input1_shared2 = (float *)(shared_buffer + 4096);
  {
    float *compute = output0;
    {
      float compute_local[4];
      float compute_local1[4];
      float compute_local2[4];

#pragma unroll
      for (int xx_c_init = 0; xx_c_init < 2; ++xx_c_init) {
        compute_local[xx_c_init] = 0.000000e+00f;
        compute_local[(xx_c_init + 2)] = 0.000000e+00f;
        compute_local1[xx_c_init] = 0.000000e+00f;
        compute_local1[(xx_c_init + 2)] = 0.000000e+00f;
        compute_local2[xx_c_init] = 0.000000e+00f;
        compute_local2[(xx_c_init + 2)] = 0.000000e+00f;
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
          input1_shared1[(
              ((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 2)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] =
              input2[(
                  ((((((int)blockIdx.z) * 1536) + (((int)threadIdx.z) * 192)) +
                    ((((((int)threadIdx.x) * 2) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >>
                      4) *
                     96)) +
                   (rc_outer * 16)) +
                  (((((int)threadIdx.x) * 2) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) &
                   15))];
          input1_shared2[(
              ((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 2)) +
              ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] =
              input3[(
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
            compute_local1[xx_c] =
                (compute_local1[xx_c] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.x) * 2)) + xx_c)] *
                  input1_shared1[((((int)threadIdx.z) * 16) + rc_inner)]));
            compute_local1[(xx_c + 2)] =
                (compute_local1[(xx_c + 2)] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.x) * 2)) + xx_c)] *
                  input1_shared1[(((((int)threadIdx.z) * 16) + rc_inner) +
                                  128)]));
            compute_local2[xx_c] =
                (compute_local2[xx_c] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.x) * 2)) + xx_c)] *
                  input1_shared2[((((int)threadIdx.z) * 16) + rc_inner)]));
            compute_local2[(xx_c + 2)] =
                (compute_local2[(xx_c + 2)] +
                 (pad_temp_shared[(
                      ((rc_inner * 32) + (((int)threadIdx.x) * 2)) + xx_c)] *
                  input1_shared2[(((((int)threadIdx.z) * 16) + rc_inner) +
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
        output1[(((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) +
                   (((int)blockIdx.y) * 32)) +
                  (((int)threadIdx.x) * 2)) +
                 xx_inner_inner_inner)] = compute_local1[xx_inner_inner_inner];
        output1[(
            (((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) +
               (((int)blockIdx.y) * 32)) +
              (((int)threadIdx.x) * 2)) +
             xx_inner_inner_inner) +
            8192)] = compute_local1[(xx_inner_inner_inner + 2)];
        output2[(((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) +
                   (((int)blockIdx.y) * 32)) +
                  (((int)threadIdx.x) * 2)) +
                 xx_inner_inner_inner)] = compute_local2[xx_inner_inner_inner];
        output2[(
            (((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) +
               (((int)blockIdx.y) * 32)) +
              (((int)threadIdx.x) * 2)) +
             xx_inner_inner_inner) +
            8192)] = compute_local2[(xx_inner_inner_inner + 2)];
      }
    }
  }
}
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_DepthwiseConv2dNative_DepthwiseConv2dNative_Convolution_Convolution_DepthwiseConv2dNative_Convolution_0(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *output0,
    float *output1, float *output2, float *output3, float *output4,
    float *output5, float *output6) {

  __shared__ char shared_buffer[5120];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_471_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    fused3_Convolution_float_float_float_cuda_Convolution_471_block_kernel1(
        input2, input5, input6, input8, output3, output4, output6, threadIdx.x,
        blockIdx.x - 64, shared_buffer);
  }
  // else if((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
  //{
  // Convolution_float_float_float_cuda_Convolution_471_block_kernel(input2,
  // input6, output4, threadIdx.x, blockIdx.x - 128, shared_buffer);
  //}
  // else if((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255)
  //{
  // Convolution_float_float_float_cuda_Convolution_471_block_kernel(input2,
  // input8, output6, threadIdx.x, blockIdx.x - 192, shared_buffer);
  //}
  else if ((int)blockIdx.x >= 256 - 128 && (int)blockIdx.x <= 1023 - 128) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_480_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 256 + 128,
        shared_buffer);
  } else if ((int)blockIdx.x >= 1024 - 128 &&
             (int)blockIdx.x <= 1024 + 767 - 128) {
    Fuse_2_DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_479_block_kernel(
        input2, input4, input7, output2, output5, threadIdx.x,
        blockIdx.x - 1024 + 128, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_DepthwiseConv2dNative_DepthwiseConv2dNative_Convolution_Convolution_DepthwiseConv2dNative_Convolution_0_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *output0,
    float *output1, float *output2, float *output3, float *output4,
    float *output5, float *output6) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_DepthwiseConv2dNative_DepthwiseConv2dNative_Convolution_Convolution_DepthwiseConv2dNative_Convolution_0<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      output0, output1, output2, output3, output4, output5, output6);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1649_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2680_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1650_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2683_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1655_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Convolution_1657_0	type: float	shape: Shape{1, 128, 8,
// 8}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1655<<<dim3(1, 4, 16), dim3(8,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_1649_0, Constant_2680_0,
// Convolution_1655_0);
// Convolution_float_float_float_cuda_Convolution_1657<<<dim3(1, 4, 16), dim3(8,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_1650_0, Constant_2683_0,
// Convolution_1657_0); Deduped function map: <src_function_name :
// deduped_function_name> Convolution_float_float_float_cuda_Convolution_1657 :
// Convolution_float_float_float_cuda_Convolution_1655

// Node name:	Convolution_1655
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1649_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2680_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1655_0	type: float	shape: Shape{1, 128, 8,
// 8}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1655_block_kernel(
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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_164(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {

  __shared__ char shared_buffer[1536];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_1655_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_1655_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_164_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_164<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2766_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1414_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Convolution_1416_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2932_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1417_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Relu_1420_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1418_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Relu_53<<<dim3(16, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1414_0, Constant_2766_0, Relu_1420_0,
// BatchNormInference_1417_0); Add_float_float_float_cuda_Add_2547<<<dim3(16, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1416_0, Constant_2932_0,
// BatchNormInference_1418_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1414_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2766_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1420_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1417_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2544<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1414_0, Constant_2766_0, BatchNormInference_1417_0);
// Relu_float_float_cuda_Relu_1420<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1417_0, Relu_1420_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Relu_53_block_kernel(
    float *input0, float *input1, float *output0, float *output1, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(16, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = relu(temp0);
  output1[tid] = temp0;
  output0[tid] = temp1;
}
// Node name:	Add_2547
// Description:	Add
// Input:
//	- name: Convolution_1416_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2932_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1418_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2547_block_kernel(float *input0, float *input1,
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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_131(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    FusedKernel_float_float_float_float_cuda_Add_Relu_53_block_kernel(
        input1, input0, output1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_2547_block_kernel(
        input2, input3, output2, threadIdx.x, blockIdx.x - 16, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_131_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {
  BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_131<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1, output2);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_1666_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2963_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2964_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1668_0	type: float	shape: Shape{1, 128, 8,
// 8}
// Output:
//	- name: BatchNormInference_1669_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: BatchNormInference_1670_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Relu_1672_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2688<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1666_0, Constant_2963_0, BatchNormInference_1669_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_69<<<dim3(16, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1668_0, Constant_2964_0, Relu_1672_0,
// BatchNormInference_1670_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	Add_2688
// Description:	Add
// Input:
//	- name: Convolution_1666_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2963_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1669_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2688_block_kernel(float *input0, float *input1,
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
//	- name: Convolution_1668_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2964_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1672_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1670_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2691<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1668_0, Constant_2964_0, BatchNormInference_1670_0);
// Relu_float_float_cuda_Relu_1672<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1670_0, Relu_1672_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Relu_69_block_kernel(
    float *input0, float *input1, float *output0, float *output1, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(16, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = relu(temp0);
  output1[tid] = temp0;
  output0[tid] = temp1;
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_167(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    Add_float_float_float_cuda_Add_2688_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31) {
    FusedKernel_float_float_float_float_cuda_Add_Relu_69_block_kernel(
        input3, input2, output2, output1, threadIdx.x, blockIdx.x - 16, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_167_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {
  BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_167<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1, output2);
}
