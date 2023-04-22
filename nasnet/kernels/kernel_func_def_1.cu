// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_196
// Description:	Constant
// Input:
// Output:
//	- name: Constant_196_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_196(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_196_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_196_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2392
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2392_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2392(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2392_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2392_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_193
// Description:	Constant
// Input:
// Output:
//	- name: Constant_193_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_193(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_193_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_193_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2731
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2731_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2731(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2731_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2731_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2482
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2482_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2482(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2482_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2482_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_183
// Description:	Constant
// Input:
// Output:
//	- name: Constant_183_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_183(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_183_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_183_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2785
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2785_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2785(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2785_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2785_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2739
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2739_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2739(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2739_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2739_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2113
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2113_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2113(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2113_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2113_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2872
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2872_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2872(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2872_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2872_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2419
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2419_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2419(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2419_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2419_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_871_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2254_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_872_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2257_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_877_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Convolution_879_0	type: float	shape: Shape{1, 32, 32,
//32}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_877<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_871_0, Constant_2254_0,
// Convolution_877_0);
// Convolution_float_float_float_cuda_Convolution_879<<<dim3(2, 16, 2), dim3(16,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_872_0, Constant_2257_0,
// Convolution_879_0); Deduped function map: <src_function_name :
// deduped_function_name> Convolution_float_float_float_cuda_Convolution_879 :
// Convolution_float_float_float_cuda_Convolution_877

// Node name:	Convolution_877
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_871_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Constant_2254_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_877_0	type: float	shape: Shape{1, 32, 32,
//32}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_877_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_52(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {

  __shared__ char shared_buffer[3072];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_877_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_877_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_52_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_52<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_742_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2777_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_744_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2744_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2836_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_746_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Relu_739_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_456_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Relu_740_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_357_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: BatchNormInference_749_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Add_756_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_747_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_748_0	type: float	shape: Shape{1,
//32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2175<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_742_0, Constant_2777_0, BatchNormInference_749_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_16<<<dim3(64, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_744_0, Constant_2744_0,
// Convolution_746_0, Constant_2836_0, Add_756_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_747<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_739_0, Constant_456_0,
// DepthwiseConv2dNative_747_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_748<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_740_0, Constant_357_0,
// DepthwiseConv2dNative_748_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	Add_2175
// Description:	Add
// Input:
//	- name: Convolution_742_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2777_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_749_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2175_block_kernel(float *input0, float *input1,
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
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_744_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2744_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_746_0	type: float	shape: Shape{1, 32, 32,
//32}
//	- name: Constant_2836_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_756_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2178<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_744_0, Constant_2744_0, BatchNormInference_750_0);
// Add_float_float_float_cuda_Add_2181<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_746_0, Constant_2836_0, BatchNormInference_751_0);
// Add_float_float_float_cuda_Add_756<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_751_0, BatchNormInference_750_0, Add_756_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_16_block_kernel(
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
// Node name:	DepthwiseConv2dNative_747
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_739_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_456_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_747_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_747_block_kernel(
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
// Node name:	DepthwiseConv2dNative_748
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_740_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_357_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_748_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_748_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_fused_kernel_DepthwiseConv2dNative_DepthwiseConv2dNative_33(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Add_float_float_float_cuda_Add_2175_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_16_block_kernel(
        input2, input3, input5, input4, output1, threadIdx.x, blockIdx.x - 64,
        NULL);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 383) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_747_block_kernel(
        input6, input7, output2, threadIdx.x, blockIdx.x - 128, NULL);
  } else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 639) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_748_block_kernel(
        input8, input9, output3, threadIdx.x, blockIdx.x - 384, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_fused_kernel_DepthwiseConv2dNative_DepthwiseConv2dNative_33_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_fused_kernel_DepthwiseConv2dNative_DepthwiseConv2dNative_33<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, input8, input9,
                                    output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: AvgPool_587_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_488_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: Relu_588_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_352_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Constant_289_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Relu_607_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_155_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Relu_605_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_260_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_606_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_153_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: Add_595_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_596_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_597_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_612_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_610_0	type: float	shape: Shape{1,
//32, 32, 32}
//	- name: DepthwiseConv2dNative_611_0	type: float	shape: Shape{1,
//32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_595<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(AvgPool_587_0, BatchNormInference_488_0, Add_595_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_596<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_588_0, Constant_352_0,
// DepthwiseConv2dNative_596_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_597<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_588_0, Constant_289_0,
// DepthwiseConv2dNative_597_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_612<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_607_0, Constant_155_0,
// DepthwiseConv2dNative_612_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_610<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_605_0, Constant_260_0,
// DepthwiseConv2dNative_610_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_611<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_606_0, Constant_153_0,
// DepthwiseConv2dNative_611_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_612 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_597
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_610 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_596
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_611 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_596

// Node name:	Add_595
// Description:	Add
// Input:
//	- name: AvgPool_587_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_488_0	type: float	shape: Shape{1,
//32, 32, 32}
// Output:
//	- name: Add_595_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void Add_float_float_float_cuda_Add_595_block_kernel(
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
// Node name:	DepthwiseConv2dNative_596
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_588_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_352_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_596_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_596_block_kernel(
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
// Node name:	DepthwiseConv2dNative_597
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_588_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_289_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_597_0	type: float	shape: Shape{1,
//32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_597_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_13(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Add_float_float_float_cuda_Add_595_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 319) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_596_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64, NULL);
  } else if ((int)blockIdx.x >= 320 && (int)blockIdx.x <= 575) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_597_block_kernel(
        input2, input4, output2, threadIdx.x, blockIdx.x - 320, NULL);
  } else if ((int)blockIdx.x >= 576 && (int)blockIdx.x <= 831) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_597_block_kernel(
        input5, input6, output3, threadIdx.x, blockIdx.x - 576, NULL);
  } else if ((int)blockIdx.x >= 832 && (int)blockIdx.x <= 1087) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_596_block_kernel(
        input7, input8, output4, threadIdx.x, blockIdx.x - 832, NULL);
  } else if ((int)blockIdx.x >= 1088 && (int)blockIdx.x <= 1343) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_596_block_kernel(
        input9, input10, output5, threadIdx.x, blockIdx.x - 1088, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_13_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_13<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4, output5);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1124_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_324_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Relu_1125_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_387_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Convolution_1136_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2899_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2896_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1132_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Convolution_1134_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2900_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: DepthwiseConv2dNative_1129_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: DepthwiseConv2dNative_1130_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: BatchNormInference_1143_0	type: float	shape: Shape{1,
//64, 16, 16}
//	- name: Add_1146_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1129<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1124_0, Constant_324_0,
// DepthwiseConv2dNative_1129_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1130<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1125_0, Constant_387_0,
// DepthwiseConv2dNative_1130_0); Add_float_float_float_cuda_Add_2394<<<dim3(32,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1136_0, Constant_2899_0,
// BatchNormInference_1143_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_37<<<dim3(32, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1132_0, Constant_2896_0,
// Convolution_1134_0, Constant_2900_0, Add_1146_0); Deduped function map:
// <src_function_name : deduped_function_name>

// Node name:	DepthwiseConv2dNative_1129
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1124_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_324_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1129_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1129_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1130
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1125_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_387_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1130_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1130_block_kernel(
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
// Node name:	Add_2394
// Description:	Add
// Input:
//	- name: Convolution_1136_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2899_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1143_0	type: float	shape: Shape{1,
//64, 16, 16}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2394_block_kernel(float *input0, float *input1,
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
//	- name: Convolution_1132_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2896_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1134_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2900_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1146_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2388<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1132_0, Constant_2896_0, BatchNormInference_1141_0);
// Add_float_float_float_cuda_Add_2391<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1134_0, Constant_2900_0, BatchNormInference_1142_0);
// Add_float_float_float_cuda_Add_1146<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1141_0, BatchNormInference_1142_0, Add_1146_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_37_block_kernel(
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
  float temp2 = add(temp0, temp1);
  output0[tid] = temp2;
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_89(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_2394_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_37_block_kernel(
        input7, input6, input8, input9, output3, threadIdx.x, blockIdx.x - 32,
        NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 191) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1129_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 64, NULL);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 319) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1130_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 192, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_89_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_89<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, input8, input9,
                                    output0, output1, output2, output3);
}
