// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	Constant_2641
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2641_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2641(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2641_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2641_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2869
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2869_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2869(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2869_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2869_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2763
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2763_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2763(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2763_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2763_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2737
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2737_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2737(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2737_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2737_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2674
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2674_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2674(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2674_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2674_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_53
// Description:	Constant
// Input:
// Output:
//	- name: Constant_53_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_53(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_53_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_53_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[256];
  bin_file.read(tmp_mem, 256);
  cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3072
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3072_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3072(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3072_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3072_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2104
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2104_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2104(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2104_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2104_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4096];
  bin_file.read(tmp_mem, 4096);
  cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2275
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2275_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2275(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2275_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2275_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_938_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_33_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Relu_930_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_141_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Relu_931_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_262_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Relu_932_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_120_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_946_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: DepthwiseConv2dNative_934_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: DepthwiseConv2dNative_935_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: DepthwiseConv2dNative_936_0	type: float	shape: Shape{1,
// 64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_946<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_938_0, Constant_33_0,
// DepthwiseConv2dNative_946_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_934<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_930_0, Constant_141_0,
// DepthwiseConv2dNative_934_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_935<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_931_0, Constant_262_0,
// DepthwiseConv2dNative_935_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_936<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_932_0, Constant_120_0,
// DepthwiseConv2dNative_936_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_935 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_946
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_936 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_946

// Node name:	DepthwiseConv2dNative_946
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_938_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_33_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_946_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_946_block_kernel(
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
// Node name:	DepthwiseConv2dNative_934
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_930_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_141_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_934_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_934_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_62(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *output0, float *output1,
    float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_946_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 255) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_934_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 128, NULL);
  } else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 383) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_946_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 256, NULL);
  } else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 511) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_946_block_kernel(
        input6, input7, output3, threadIdx.x, blockIdx.x - 384, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_62_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *output0, float *output1,
    float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_62<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, output0, output1,
                                    output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_549_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_44_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_550_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_136_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Convolution_555_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Constant_2897_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2810_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_553_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Constant_2811_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_557_0	type: float	shape: Shape{1, 32, 32,
// 32}
// Output:
//	- name: DepthwiseConv2dNative_558_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: DepthwiseConv2dNative_559_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: BatchNormInference_561_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Add_567_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_558<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_549_0, Constant_44_0,
// DepthwiseConv2dNative_558_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_559<<<dim3(256,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_550_0, Constant_136_0,
// DepthwiseConv2dNative_559_0); Add_float_float_float_cuda_Add_2070<<<dim3(64,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_555_0, Constant_2897_0,
// BatchNormInference_561_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_4<<<dim3(64, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_553_0, Constant_2810_0,
// Convolution_557_0, Constant_2811_0, Add_567_0); Deduped function map:
// <src_function_name : deduped_function_name>

// Node name:	DepthwiseConv2dNative_558
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_549_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_44_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_558_0	type: float	shape: Shape{1,
// 32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_558_block_kernel(
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
// Node name:	DepthwiseConv2dNative_559
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_550_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_136_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_559_0	type: float	shape: Shape{1,
// 32, 32, 32}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_559_block_kernel(
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
// Node name:	Add_2070
// Description:	Add
// Input:
//	- name: Convolution_555_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Constant_2897_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_561_0	type: float	shape: Shape{1,
// 32, 32, 32}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2070_block_kernel(float *input0, float *input1,
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
//	- name: Convolution_553_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Constant_2810_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_557_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Constant_2811_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_567_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2067<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_553_0, Constant_2810_0, BatchNormInference_560_0);
// Add_float_float_float_cuda_Add_2073<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_557_0, Constant_2811_0, BatchNormInference_562_0);
// Add_float_float_float_cuda_Add_567<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_562_0, BatchNormInference_560_0, Add_567_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_4_block_kernel(
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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_6(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Add_float_float_float_cuda_Add_2070_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_4_block_kernel(
        input7, input6, input9, input8, output3, threadIdx.x, blockIdx.x - 64,
        NULL);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 383) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_558_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 128, NULL);
  } else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 639) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_559_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 384, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_6_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_6<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, input8, input9,
                                    output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1250_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_416_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Relu_1251_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_443_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Convolution_1262_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2915_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2791_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1258_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2914_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1260_0	type: float	shape: Shape{1, 64, 16,
// 16}
// Output:
//	- name: DepthwiseConv2dNative_1255_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: DepthwiseConv2dNative_1256_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: BatchNormInference_1269_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Add_1272_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1255<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1250_0, Constant_416_0,
// DepthwiseConv2dNative_1255_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1256<<<dim3(128,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1251_0, Constant_443_0,
// DepthwiseConv2dNative_1256_0); Add_float_float_float_cuda_Add_2466<<<dim3(32,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1262_0, Constant_2915_0,
// BatchNormInference_1269_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_45<<<dim3(32, 1,
// 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1258_0, Constant_2791_0,
// Convolution_1260_0, Constant_2914_0, Add_1272_0); Deduped function map:
// <src_function_name : deduped_function_name>

// Node name:	DepthwiseConv2dNative_1255
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1250_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_416_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1255_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1255_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1256
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1251_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_443_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1256_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1256_block_kernel(
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
// Node name:	Add_2466
// Description:	Add
// Input:
//	- name: Convolution_1262_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2915_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1269_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2466_block_kernel(float *input0, float *input1,
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
//	- name: Convolution_1258_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2791_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1260_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2914_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1272_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2460<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1258_0, Constant_2791_0, BatchNormInference_1267_0);
// Add_float_float_float_cuda_Add_2463<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1260_0, Constant_2914_0, BatchNormInference_1268_0);
// Add_float_float_float_cuda_Add_1272<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1268_0, BatchNormInference_1267_0, Add_1272_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_45_block_kernel(
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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_107(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_2466_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_45_block_kernel(
        input7, input6, input9, input8, output3, threadIdx.x, blockIdx.x - 32,
        NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 191) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1255_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 64, NULL);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 319) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1256_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 192, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_107_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_107<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, input8, input9,
                                    output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1635_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_14_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Relu_1633_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_429_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Relu_1634_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_305_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: AvgPool_1615_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1544_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Relu_1616_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_378_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Constant_19_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1640_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1638_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1639_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Add_1623_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1624_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1625_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1640<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1635_0, Constant_14_0,
// DepthwiseConv2dNative_1640_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1633_0, Constant_429_0,
// DepthwiseConv2dNative_1638_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1639<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1634_0, Constant_305_0,
// DepthwiseConv2dNative_1639_0); Add_float_float_float_cuda_Add_1623<<<dim3(16,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1615_0, BatchNormInference_1544_0,
// Add_1623_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1624<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1616_0, Constant_378_0,
// DepthwiseConv2dNative_1624_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1625<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1616_0, Constant_19_0,
// DepthwiseConv2dNative_1625_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1639 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1624 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1625 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1640

// Node name:	DepthwiseConv2dNative_1640
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1635_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_14_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1640_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1640_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1638
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1633_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_429_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1638_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638_block_kernel(
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
// Node name:	Add_1623
// Description:	Add
// Input:
//	- name: AvgPool_1615_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1544_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Output:
//	- name: Add_1623_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_1623_block_kernel(float *input0, float *input1,
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_161(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    Add_float_float_float_cuda_Add_1623_block_kernel(
        input6, input7, output3, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 79) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1640_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 16, NULL);
  } else if ((int)blockIdx.x >= 80 && (int)blockIdx.x <= 143) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 80, NULL);
  } else if ((int)blockIdx.x >= 144 && (int)blockIdx.x <= 207) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 144, NULL);
  } else if ((int)blockIdx.x >= 208 && (int)blockIdx.x <= 271) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638_block_kernel(
        input8, input9, output4, threadIdx.x, blockIdx.x - 208, NULL);
  } else if ((int)blockIdx.x >= 272 && (int)blockIdx.x <= 335) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1640_block_kernel(
        input8, input10, output5, threadIdx.x, blockIdx.x - 272, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_161_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *input10, float *output0, float *output1, float *output2,
    float *output3, float *output4, float *output5) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_161<<<
      grids, blocks, mem, stream>>>(
      input0, input1, input2, input3, input4, input5, input6, input7, input8,
      input9, input10, output0, output1, output2, output3, output4, output5);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_1215_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2910_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2911_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1217_0	type: float	shape: Shape{1, 64, 16,
// 16}
// Output:
//	- name: BatchNormInference_1218_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: BatchNormInference_1219_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Relu_1221_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2439<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1215_0, Constant_2910_0, BatchNormInference_1218_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_44<<<dim3(32, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1217_0, Constant_2911_0, Relu_1221_0,
// BatchNormInference_1219_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	Add_2439
// Description:	Add
// Input:
//	- name: Convolution_1215_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2910_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1218_0	type: float	shape: Shape{1,
// 64, 16, 16}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2439_block_kernel(float *input0, float *input1,
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
//	- name: Convolution_1217_0	type: float	shape: Shape{1, 64, 16,
// 16}
//	- name: Constant_2911_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1221_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1219_0	type: float	shape: Shape{1,
// 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2442<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1217_0, Constant_2911_0, BatchNormInference_1219_0);
// Relu_float_float_cuda_Relu_1221<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1219_0, Relu_1221_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Relu_44_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_102(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Add_float_float_float_cuda_Add_2439_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    FusedKernel_float_float_float_float_cuda_Add_Relu_44_block_kernel(
        input3, input2, output2, output1, threadIdx.x, blockIdx.x - 32, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_102_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {
  BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_102<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1, output2);
}
// Node name:	Constant_2368
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2368_0	type: float	shape: Shape{64, 384, 1, 1}
void Constant_float_cuda_Constant_2368(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2368_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2368_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[98304];
  bin_file.read(tmp_mem, 98304);
  cudaMemcpyAsync(output0, tmp_mem, 98304, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
