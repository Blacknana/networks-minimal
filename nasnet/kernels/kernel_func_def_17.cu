// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	Constant_2362
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2362_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2362(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2362_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2362_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2805
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2805_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2805(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2805_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2805_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_74
// Description:	Constant
// Input:
// Output:
//	- name: Constant_74_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_74(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_74_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_74_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[3200];
  bin_file.read(tmp_mem, 3200);
  cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_67
// Description:	Constant
// Input:
// Output:
//	- name: Constant_67_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_67(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_67_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_67_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3170
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3170_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3170(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3170_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3170_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_457
// Description:	Constant
// Input:
// Output:
//	- name: Constant_457_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_457(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_457_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_457_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_307
// Description:	Constant
// Input:
// Output:
//	- name: Constant_307_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_307(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_307_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_307_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3032
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3032_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_3032(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3032_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3032_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_121
// Description:	Constant
// Input:
// Output:
//	- name: Constant_121_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_121(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_121_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_121_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Convolution_466
// Description:	Convolution
// Input:
//	- name: Reshape_464_0	type: float	shape: Shape{1, 3, 32, 32}
//	- name: Constant_2011_0	type: float	shape: Shape{96, 3, 3, 3}
// Output:
//	- name: Convolution_466_0	type: float	shape: Shape{1, 96, 32,
//32}
extern "C" __global__ void
Convolution_float_float_float_cuda_Convolution_466(float *input0, float *input1,
                                                   float *output0) {
  __shared__ float pad_temp_shared[408];
  __shared__ float input1_shared[864];
  {
    float *compute = output0;
    {
      float compute_local[4];

#pragma unroll
      for (int yy_c_init = 0; yy_c_init < 2; ++yy_c_init) {
        compute_local[yy_c_init] = 0.000000e+00f;
        compute_local[(yy_c_init + 2)] = 0.000000e+00f;
      }
      if (((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) < 408) {
        if (((int)threadIdx.x) < 26) {
          pad_temp_shared[((((int)threadIdx.z) * 26) + ((int)threadIdx.x))] =
              (((((1 <=
                   ((((int)blockIdx.y) * 2) +
                    ((((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) % 136) /
                     34))) &&
                  (((((int)blockIdx.y) * 2) +
                    ((((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) % 136) /
                     34)) < 33)) &&
                 (1 <=
                  (((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) % 34))) &&
                ((((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) % 34) < 33))
                   ? input0[(
                         (((((((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) /
                              136) *
                             1024) +
                            (((int)blockIdx.y) * 64)) +
                           (((((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) %
                              136) /
                             34) *
                            32)) +
                          (((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) %
                           34)) -
                         33)]
                   : 0.000000e+00f);
        }
      }
#pragma unroll
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0;
           ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2;
           ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
        if (((((int)threadIdx.z) * 2) +
             (((((int)threadIdx.x) * 2) +
               ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) /
              27)) < 32) {
          if (((((int)threadIdx.z) * 6) +
               (((((int)threadIdx.x) * 2) +
                 ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) /
                9)) < 96) {
            if (((((int)threadIdx.z) * 18) +
                 (((((int)threadIdx.x) * 2) +
                   ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) /
                  3)) < 288) {
              if ((((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 2)) +
                   ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 864) {
                if (((((int)threadIdx.x) * 2) +
                     ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) <
                    54) {
                  input1_shared[(
                      ((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 2)) +
                      ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] =
                      input1[(
                          (((((int)blockIdx.z) * 864) +
                            (((int)threadIdx.z) * 54)) +
                           (((int)threadIdx.x) * 2)) +
                          ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)];
                }
              }
            }
          }
        }
      }
      __syncthreads();
#pragma unroll
      for (int rc_inner = 0; rc_inner < 3; ++rc_inner) {
#pragma unroll
        for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
#pragma unroll
          for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
#pragma unroll
            for (int yy_c = 0; yy_c < 2; ++yy_c) {
              compute_local[yy_c] =
                  (compute_local[yy_c] +
                   (pad_temp_shared[(
                        ((((rc_inner * 136) + (yy_c * 34)) + (ry_inner * 34)) +
                         ((int)threadIdx.x)) +
                        rx_inner)] *
                    input1_shared[(
                        (((((int)threadIdx.z) * 27) + (rc_inner * 9)) +
                         (ry_inner * 3)) +
                        rx_inner)]));
              compute_local[(yy_c + 2)] =
                  (compute_local[(yy_c + 2)] +
                   (pad_temp_shared[(
                        ((((rc_inner * 136) + (yy_c * 34)) + (ry_inner * 34)) +
                         ((int)threadIdx.x)) +
                        rx_inner)] *
                    input1_shared[(
                        ((((((int)threadIdx.z) * 27) + (rc_inner * 9)) +
                          (ry_inner * 3)) +
                         rx_inner) +
                        432)]));
            }
          }
        }
      }
#pragma unroll
      for (int yy_inner_inner_inner = 0; yy_inner_inner_inner < 2;
           ++yy_inner_inner_inner) {
        compute[(((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 1024)) +
                   (((int)blockIdx.y) * 64)) +
                  (yy_inner_inner_inner * 32)) +
                 ((int)threadIdx.x))] = compute_local[yy_inner_inner_inner];
        compute[(
            (((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 1024)) +
               (((int)blockIdx.y) * 64)) +
              (yy_inner_inner_inner * 32)) +
             ((int)threadIdx.x)) +
            16384)] = compute_local[(yy_inner_inner_inner + 2)];
      }
    }
  }
}
extern void Convolution_float_float_float_cuda_Convolution_466_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *output0) {
  Convolution_float_float_float_cuda_Convolution_466<<<grids, blocks, mem,
                                                       stream>>>(input0, input1,
                                                                 output0);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1509_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_99_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Relu_1507_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_111_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Relu_1508_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_252_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: AvgPool_1489_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1417_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Relu_1490_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_259_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Constant_223_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1514_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: DepthwiseConv2dNative_1512_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: DepthwiseConv2dNative_1513_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Add_1497_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1498_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: DepthwiseConv2dNative_1499_0	type: float	shape: Shape{1,
//128, 8, 8}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1514<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1509_0, Constant_99_0,
// DepthwiseConv2dNative_1514_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1512<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1507_0, Constant_111_0,
// DepthwiseConv2dNative_1512_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1513<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1508_0, Constant_252_0,
// DepthwiseConv2dNative_1513_0); Add_float_float_float_cuda_Add_1497<<<dim3(16,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1489_0, BatchNormInference_1417_0,
// Add_1497_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1498<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1490_0, Constant_259_0,
// DepthwiseConv2dNative_1498_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1499<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1490_0, Constant_223_0,
// DepthwiseConv2dNative_1499_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1512 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1514
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1498 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1513
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1499 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1514

// Node name:	DepthwiseConv2dNative_1514
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1509_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_99_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1514_0	type: float	shape: Shape{1,
//128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1514_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1513
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1508_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_252_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1513_0	type: float	shape: Shape{1,
//128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1513_block_kernel(
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
// Node name:	Add_1497
// Description:	Add
// Input:
//	- name: AvgPool_1489_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1417_0	type: float	shape: Shape{1,
//128, 8, 8}
// Output:
//	- name: Add_1497_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_1497_block_kernel(float *input0, float *input1,
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_143(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    Add_float_float_float_cuda_Add_1497_block_kernel(
        input6, input7, output3, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 79) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1514_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 16, NULL);
  } else if ((int)blockIdx.x >= 80 && (int)blockIdx.x <= 143) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1514_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 80, NULL);
  } else if ((int)blockIdx.x >= 144 && (int)blockIdx.x <= 207) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1513_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 144, NULL);
  } else if ((int)blockIdx.x >= 208 && (int)blockIdx.x <= 271) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1513_block_kernel(
        input8, input9, output4, threadIdx.x, blockIdx.x - 208, NULL);
  } else if ((int)blockIdx.x >= 272 && (int)blockIdx.x <= 335) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1514_block_kernel(
        input8, input10, output5, threadIdx.x, blockIdx.x - 272, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_143_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_143<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4, output5);
}
// Node name:	 BlockFusion
// Input:
//	- name: Slice_1545_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1550_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2620_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3150_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1552_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2626_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3154_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: AvgPool_1547_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1551_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2623_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3152_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1549_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1572_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1574_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Add_1553_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1573_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Relu_float_float_cuda_Relu_1549<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Slice_1545_0, Relu_1549_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3149<<<dim3(1,
// 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1550_0, Constant_2620_0,
// Constant_3150_0, Relu_1572_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3153<<<dim3(1,
// 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1552_0, Constant_2626_0,
// Constant_3154_0, Relu_1574_0); Add_float_float_float_cuda_Add_1553<<<dim3(16,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1547_0, AvgPool_1547_0, Add_1553_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3151<<<dim3(1,
// 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1551_0, Constant_2623_0,
// Constant_3152_0, Relu_1573_0); Deduped function map: <src_function_name :
// deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3153 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3149
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3151 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3149

// Node name:	Relu_1549
// Description:	Relu
// Input:
//	- name: Slice_1545_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1549_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Relu_float_float_cuda_Relu_1549_block_kernel(float *input0, float *output0,
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
// Node name:	Matched_Pattern_3149
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1550_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: Constant_2620_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3150_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1572_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3149_block_kernel(
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
// Node name:	Add_1553
// Description:	Add
// Input:
//	- name: AvgPool_1547_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: AvgPool_1547_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1553_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_1553_block_kernel(float *input0, float *input1,
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Matched_Pattern_Matched_Pattern_Add_Matched_Pattern_151(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {

  __shared__ char shared_buffer[1536];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    Relu_float_float_cuda_Relu_1549_block_kernel(input0, output0, threadIdx.x,
                                                 blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_1553_block_kernel(
        input7, input7, output3, threadIdx.x, blockIdx.x - 16, shared_buffer);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 95) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3149_block_kernel(
        input1, input2, input3, output1, threadIdx.x, blockIdx.x - 32,
        shared_buffer);
  } else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 159) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3149_block_kernel(
        input4, input5, input6, output2, threadIdx.x, blockIdx.x - 96,
        shared_buffer);
  } else if ((int)blockIdx.x >= 160 && (int)blockIdx.x <= 223) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3149_block_kernel(
        input8, input9, input10, output4, threadIdx.x, blockIdx.x - 160,
        shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Matched_Pattern_Matched_Pattern_Add_Matched_Pattern_151_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Matched_Pattern_Matched_Pattern_Add_Matched_Pattern_151<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: BatchNormInference_767_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Relu_771_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_356_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Constant_68_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Constant_235_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: Slice_769_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_775_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_777_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_776_0	type: float	shape: Shape{1,
//32, 32, 32}
// Fused functions:
// Slice_float_float_cuda_Slice_769<<<dim3(512, 1, 1), dim3(64, 1, 1), 0,
// 0>>>(BatchNormInference_767_0, Slice_769_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_775<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_771_0, Constant_356_0,
// DepthwiseConv2dNative_775_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_777<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_771_0, Constant_68_0,
// DepthwiseConv2dNative_777_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_776<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_771_0, Constant_235_0,
// DepthwiseConv2dNative_776_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_776 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_777

// Node name:	Slice_769
// Description:	Slice
// Input:
//	- name: BatchNormInference_767_0	type: float	shape: Shape{1,
//32, 32, 32}
// Output:
//	- name: Slice_769_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void
Slice_float_float_cuda_Slice_769_block_kernel(float *input0, float *output0,
                                              int thread_id, int block_id,
                                              char *shared_buffer) {
  if (thread_id >= 64) {
    return;
  }
  const dim3 blockDim(64, 1, 1);
  const dim3 gridDim(512, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < 32768) {
    uint32_t input_strides[] = {32768, 1024, 32, 1};
    uint32_t output_strides[] = {32768, 1024, 32, 1};
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
// Node name:	DepthwiseConv2dNative_775
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_771_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_356_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_775_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_775_block_kernel(
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
// Node name:	DepthwiseConv2dNative_777
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_771_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_68_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_777_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_777_block_kernel(
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
__device__ __forceinline__ static void
fused2_DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_777_block_kernel(
    float *input0, float *input1, float *input2, float *output0, float *output1,
    int thread_id, int block_id, char *shared_buffer) {
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
    S sum2 = static_cast<S>(0);
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
          sum2 += static_cast<S>(__ldg(input + input_offset)) *
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
            sum += static_cast<S>(__ldg(input + input_offset)) *
                   static_cast<S>(__ldg(filter + filter_offset));
            sum2 += static_cast<S>(__ldg(input + input_offset)) *
                    static_cast<S>(__ldg(input2 + filter_offset));
          }
        }
      }
    }

    output[thread_id] = static_cast<S>(sum);
    output1[thread_id] = static_cast<S>(sum2);
  }
}
extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_Slice_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_38(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 255) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_775_block_kernel(
        input1, input2, output1, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 511) {
    fused2_DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_777_block_kernel(
        input1, input3, input4, output2, output3, threadIdx.x, blockIdx.x - 256,
        NULL);
  }
  // else if((int)blockIdx.x >= 512 && (int)blockIdx.x <= 767)
  //{
  //    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_777_block_kernel(input1,
  //    input4, output3, threadIdx.x, blockIdx.x - 512, NULL);
  //}
  else if ((int)blockIdx.x >= 768 - 256 && (int)blockIdx.x <= 1279 - 256) {
    Slice_float_float_cuda_Slice_769_block_kernel(input0, output0, threadIdx.x,
                                                  blockIdx.x - 768 + 256, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_Slice_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_38_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_Slice_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_38<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_863_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_463_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Relu_864_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_94_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_871_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_872_0	type: float	shape: Shape{1,
//32, 32, 32}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_871<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_863_0, Constant_463_0,
// DepthwiseConv2dNative_871_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_872<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_864_0, Constant_94_0,
// DepthwiseConv2dNative_872_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	DepthwiseConv2dNative_871
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_863_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_463_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_871_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_871_block_kernel(
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
// Node name:	DepthwiseConv2dNative_872
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_864_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_94_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_872_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_872_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_51(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 255) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_871_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 511) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_872_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 256, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_51_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_51<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2869_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1389_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2931_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1391_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Convolution_1393_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2746_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1405_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_441_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: Add_1401_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1398_0	type: float	shape: Shape{1,
//128, 8, 8}
//	- name: DepthwiseConv2dNative_1406_0	type: float	shape: Shape{1,
//128, 8, 8}
// Fused functions:
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_50<<<dim3(16, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1389_0, Constant_2869_0,
// Convolution_1391_0, Constant_2931_0, Add_1401_0);
// Add_float_float_float_cuda_Add_2535<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1393_0, Constant_2746_0, BatchNormInference_1398_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1406<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1405_0, Constant_441_0,
// DepthwiseConv2dNative_1406_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1389_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2869_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1391_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2931_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1401_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2529<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1389_0, Constant_2869_0, BatchNormInference_1396_0);
// Add_float_float_float_cuda_Add_2532<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1391_0, Constant_2931_0, BatchNormInference_1397_0);
// Add_float_float_float_cuda_Add_1401<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1396_0, BatchNormInference_1397_0, Add_1401_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_50_block_kernel(
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
  float temp2 = add(temp0, temp1);
  output0[tid] = temp2;
}
// Node name:	Add_2535
// Description:	Add
// Input:
//	- name: Convolution_1393_0	type: float	shape: Shape{1, 128, 8,
//8}
//	- name: Constant_2746_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1398_0	type: float	shape: Shape{1,
//128, 8, 8}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2535_block_kernel(float *input0, float *input1,
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
// Node name:	DepthwiseConv2dNative_1406
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1405_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_441_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1406_0	type: float	shape: Shape{1,
//128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1406_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_129(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *output0, float *output1,
    float *output2) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_50_block_kernel(
        input1, input0, input3, input2, output0, threadIdx.x, blockIdx.x - 0,
        NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_2535_block_kernel(
        input4, input5, output1, threadIdx.x, blockIdx.x - 16, NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 95) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1406_block_kernel(
        input6, input7, output2, threadIdx.x, blockIdx.x - 32, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_129_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *output0, float *output1,
    float *output2) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_129<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, output0, output1,
                                    output2);
}
