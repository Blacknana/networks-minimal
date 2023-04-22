// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1380_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_129_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Relu_1381_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_244_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Relu_1382_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_88_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Relu_1395_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_189_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1384_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1385_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1386_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: DepthwiseConv2dNative_1400_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1384<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1380_0, Constant_129_0,
// DepthwiseConv2dNative_1384_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1385<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1381_0, Constant_244_0,
// DepthwiseConv2dNative_1385_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1386<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1382_0, Constant_88_0,
// DepthwiseConv2dNative_1386_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1400<<<dim3(64,
// 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1395_0, Constant_189_0,
// DepthwiseConv2dNative_1400_0); Deduped function map: <src_function_name :
// deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1386 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1385
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1400 :
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1385

// Node name:	DepthwiseConv2dNative_1384
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1380_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_129_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1384_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1384_block_kernel(
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
// Node name:	DepthwiseConv2dNative_1385
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1381_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_244_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1385_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1385_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_127(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *output0, float *output1,
    float *output2, float *output3) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1384_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1385_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64, NULL);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1385_block_kernel(
        input4, input5, output2, threadIdx.x, blockIdx.x - 128, NULL);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255) {
    DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1385_block_kernel(
        input6, input7, output3, threadIdx.x, blockIdx.x - 192, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_127_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *output0, float *output1,
    float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_127<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, output0, output1,
                                    output2, output3);
}
// Node name:	Constant_2844
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2844_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2844(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2844_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2844_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2668
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2668_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2668(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2668_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2668_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2467
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2467_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2467(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2467_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2467_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_159
// Description:	Constant
// Input:
// Output:
//	- name: Constant_159_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_159(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_159_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_159_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2907
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2907_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2907(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2907_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2907_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2746
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2746_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2746(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2746_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2746_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_169
// Description:	Constant
// Input:
// Output:
//	- name: Constant_169_0	type: float	shape: Shape{7, 7, 128, 1}
void Constant_float_cuda_Constant_169(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_169_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_169_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[25088];
  bin_file.read(tmp_mem, 25088);
  cudaMemcpyAsync(output0, tmp_mem, 25088, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_385
// Description:	Constant
// Input:
// Output:
//	- name: Constant_385_0	type: float	shape: Shape{7, 7, 64, 1}
void Constant_float_cuda_Constant_385(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_385_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_385_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12544];
  bin_file.read(tmp_mem, 12544);
  cudaMemcpyAsync(output0, tmp_mem, 12544, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2323
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2323_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2323(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2323_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2323_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_836_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2236_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3038_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_834_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2230_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3040_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_835_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2233_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3042_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_831_0	type: float	shape: Shape{1,
// 32, 32, 32}
// Output:
//	- name: Relu_857_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_855_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_856_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Slice_833_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3037<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_836_0, Constant_2236_0,
// Constant_3038_0, Relu_857_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3039<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_834_0, Constant_2230_0,
// Constant_3040_0, Relu_855_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3041<<<dim3(1,
// 16, 4), dim3(8, 1, 16), 0, 0>>>(DepthwiseConv2dNative_835_0, Constant_2233_0,
// Constant_3042_0, Relu_856_0); Slice_float_float_cuda_Slice_833<<<dim3(512, 1,
// 1), dim3(64, 1, 1), 0, 0>>>(BatchNormInference_831_0, Slice_833_0); Deduped
// function map: <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3039 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3037
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3041 :
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3037

// Node name:	Matched_Pattern_3037
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_836_0	type: float	shape: Shape{1,
// 64, 16, 16}
//	- name: Constant_2236_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3038_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_857_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __forceinline__ void
Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3037_block_kernel(
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
// Node name:	Slice_833
// Description:	Slice
// Input:
//	- name: BatchNormInference_831_0	type: float	shape: Shape{1,
// 32, 32, 32}
// Output:
//	- name: Slice_833_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __forceinline__ void
Slice_float_float_cuda_Slice_833_block_kernel(float *input0, float *output0,
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

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Slice_47(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {

  __shared__ char shared_buffer[2048];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3037_block_kernel(
        input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0,
        shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3037_block_kernel(
        input3, input4, input5, output1, threadIdx.x, blockIdx.x - 64,
        shared_buffer);
  } else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191) {
    Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3037_block_kernel(
        input6, input7, input8, output2, threadIdx.x, blockIdx.x - 128,
        shared_buffer);
  } else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 703) {
    Slice_float_float_cuda_Slice_833_block_kernel(
        input9, output3, threadIdx.x, blockIdx.x - 192, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Slice_47_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *input6, float *input7, float *input8, float *input9,
    float *output0, float *output1, float *output2, float *output3) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Slice_47<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, input6, input7, input8, input9,
                                    output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1664_0	type: float	shape: Shape{1, 768, 8, 8}
//	- name: Constant_2686_0	type: float	shape: Shape{128, 768, 1, 1}
//	- name: Constant_2689_0	type: float	shape: Shape{128, 768, 1, 1}
// Output:
//	- name: Convolution_1666_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Convolution_1668_0	type: float	shape: Shape{1, 128, 8,
// 8}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1666<<<dim3(2, 2, 8), dim3(4,
// 4, 16), 0, 0>>>(Relu_1664_0, Constant_2686_0, Convolution_1666_0);
// Convolution_float_float_float_cuda_Convolution_1668<<<dim3(2, 2, 8), dim3(4,
// 4, 16), 0, 0>>>(Relu_1664_0, Constant_2689_0, Convolution_1668_0); Deduped
// function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1668 :
// Convolution_float_float_float_cuda_Convolution_1666

// Node name:	Convolution_1666
// Description:	Convolution
// Input:
//	- name: Relu_1664_0	type: float	shape: Shape{1, 768, 8, 8}
//	- name: Constant_2686_0	type: float	shape: Shape{128, 768, 1, 1}
// Output:
//	- name: Convolution_1666_0	type: float	shape: Shape{1, 128, 8,
// 8}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1666_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(4, 4, 16);
  const dim3 gridDim(2, 2, 8);
  const dim3 threadIdx(thread_id % 4, thread_id / 4 % 4, thread_id / 16);
  const dim3 blockIdx(block_id % 2, block_id / 2 % 2, block_id / 4);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 2048);
  {
    float *compute = output0;
    {
      float compute_local[1];

      compute_local[0] = 0.000000e+00f;
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              (((((((int)threadIdx.z) * 128) +
                  ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                    2) *
                   64)) +
                 (((int)blockIdx.y) * 32)) +
                ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                 8)) +
               (((int)blockIdx.x) * 4)) +
              ((((int)threadIdx.x) & 1) * 2))]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[((((((((int)threadIdx.z) * 128) +
                           ((((((int)threadIdx.y) * 2) +
                              (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                             2) *
                            64)) +
                          (((int)blockIdx.y) * 32)) +
                         ((((((int)threadIdx.y) * 2) +
                            (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                           3) *
                          8)) +
                        (((int)blockIdx.x) * 4)) +
                       (((((int)threadIdx.x) * 2) + 1) & 3))]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                   (((int)threadIdx.y) * 8)) +
                  (((int)threadIdx.x) * 2))];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              2048)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       2048)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  32)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  33)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              4096)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       4096)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  64)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  65)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              6144)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       6144)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  96)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  97)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              8192)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       8192)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  128)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  129)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              10240)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       10240)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  160)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  161)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              12288)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       12288)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  192)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  193)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              14336)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       14336)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  224)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  225)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              16384)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       16384)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  256)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  257)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              18432)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       18432)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  288)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  289)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              20480)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       20480)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  320)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  321)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              22528)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       22528)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  352)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  353)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              24576)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       24576)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  384)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  385)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              26624)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       26624)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  416)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  417)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              28672)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       28672)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  448)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  449)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              30720)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       30720)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  480)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  481)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              32768)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       32768)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  512)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  513)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              34816)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       34816)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  544)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  545)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              36864)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       36864)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  576)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  577)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              38912)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       38912)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  608)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  609)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              40960)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       40960)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  640)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  641)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              43008)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       43008)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  672)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  673)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              45056)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       45056)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  704)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  705)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              47104)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       47104)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  736)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  737)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute[((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                  (((int)blockIdx.y) * 32)) +
                 (((int)threadIdx.y) * 8)) +
                (((int)blockIdx.x) * 4)) +
               ((int)threadIdx.x))] = compute_local[0];
    }
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_166(
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {

  __shared__ char shared_buffer[4096];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Convolution_float_float_float_cuda_Convolution_1666_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_1666_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 32, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_166_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_166<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1706_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2716_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1707_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2719_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1715_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Convolution_1717_0	type: float	shape: Shape{1, 128, 8,
// 8}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1715<<<dim3(1, 4, 16), dim3(8,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_1706_0, Constant_2716_0,
// Convolution_1715_0);
// Convolution_float_float_float_cuda_Convolution_1717<<<dim3(1, 4, 16), dim3(8,
// 2, 8), 0, 0>>>(DepthwiseConv2dNative_1707_0, Constant_2719_0,
// Convolution_1717_0); Deduped function map: <src_function_name :
// deduped_function_name> Convolution_float_float_float_cuda_Convolution_1717 :
// Convolution_float_float_float_cuda_Convolution_1715

// Node name:	Convolution_1715
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1706_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Constant_2716_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1715_0	type: float	shape: Shape{1, 128, 8,
// 8}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1715_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_173(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {

  __shared__ char shared_buffer[1536];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_1715_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    Convolution_float_float_float_cuda_Convolution_1715_block_kernel(
        input2, input3, output1, threadIdx.x, blockIdx.x - 64, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_173_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_173<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_1540_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2943_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1542_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2946_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1543_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: BatchNormInference_1544_0	type: float	shape: Shape{1,
// 128, 8, 8}
//	- name: Relu_1546_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2616<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1540_0, Constant_2943_0, BatchNormInference_1543_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_61<<<dim3(16, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1542_0, Constant_2946_0, Relu_1546_0,
// BatchNormInference_1544_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	Add_2616
// Description:	Add
// Input:
//	- name: Convolution_1540_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2943_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1543_0	type: float	shape: Shape{1,
// 128, 8, 8}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2616_block_kernel(float *input0, float *input1,
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
//	- name: Convolution_1542_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Constant_2946_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1546_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1544_0	type: float	shape: Shape{1,
// 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2619<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1542_0, Constant_2946_0, BatchNormInference_1544_0);
// Relu_float_float_cuda_Relu_1546<<<dim3(16, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1544_0, Relu_1546_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Relu_61_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_149(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15) {
    Add_float_float_float_cuda_Add_2616_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31) {
    FusedKernel_float_float_float_float_cuda_Add_Relu_61_block_kernel(
        input2, input3, output2, output1, threadIdx.x, blockIdx.x - 16, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_149_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {
  BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_149<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1, output2);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1538_0	type: float	shape: Shape{1, 768, 8, 8}
//	- name: Constant_2614_0	type: float	shape: Shape{128, 768, 1, 1}
//	- name: Constant_2617_0	type: float	shape: Shape{128, 768, 1, 1}
// Output:
//	- name: Convolution_1540_0	type: float	shape: Shape{1, 128, 8,
// 8}
//	- name: Convolution_1542_0	type: float	shape: Shape{1, 128, 8,
// 8}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1540<<<dim3(2, 2, 8), dim3(4,
// 4, 16), 0, 0>>>(Relu_1538_0, Constant_2614_0, Convolution_1540_0);
// Convolution_float_float_float_cuda_Convolution_1542<<<dim3(2, 2, 8), dim3(4,
// 4, 16), 0, 0>>>(Relu_1538_0, Constant_2617_0, Convolution_1542_0); Deduped
// function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1542 :
// Convolution_float_float_float_cuda_Convolution_1540

// Node name:	Convolution_1540
// Description:	Convolution
// Input:
//	- name: Relu_1538_0	type: float	shape: Shape{1, 768, 8, 8}
//	- name: Constant_2614_0	type: float	shape: Shape{128, 768, 1, 1}
// Output:
//	- name: Convolution_1540_0	type: float	shape: Shape{1, 128, 8,
// 8}
__device__ __forceinline__ void
Convolution_float_float_float_cuda_Convolution_1540_block_kernel(
    float *input0, float *input1, float *output0, int thread_id, int block_id,
    char *shared_buffer) {
  if (thread_id >= 256) {
    return;
  }
  const dim3 blockDim(4, 4, 16);
  const dim3 gridDim(2, 2, 8);
  const dim3 threadIdx(thread_id % 4, thread_id / 4 % 4, thread_id / 16);
  const dim3 blockIdx(block_id % 2, block_id / 2 % 2, block_id / 4);
  float *pad_temp_shared = (float *)(shared_buffer + 0);
  float *input1_shared = (float *)(shared_buffer + 2048);
  {
    float *compute = output0;
    {
      float compute_local[1];

      compute_local[0] = 0.000000e+00f;
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              (((((((int)threadIdx.z) * 128) +
                  ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                    2) *
                   64)) +
                 (((int)blockIdx.y) * 32)) +
                ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                 8)) +
               (((int)blockIdx.x) * 4)) +
              ((((int)threadIdx.x) & 1) * 2))]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[((((((((int)threadIdx.z) * 128) +
                           ((((((int)threadIdx.y) * 2) +
                              (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                             2) *
                            64)) +
                          (((int)blockIdx.y) * 32)) +
                         ((((((int)threadIdx.y) * 2) +
                            (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                           3) *
                          8)) +
                        (((int)blockIdx.x) * 4)) +
                       (((((int)threadIdx.x) * 2) + 1) & 3))]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                   (((int)threadIdx.y) * 8)) +
                  (((int)threadIdx.x) * 2))];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  1)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              2048)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       2048)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  32)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  33)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              4096)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       4096)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  64)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  65)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              6144)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       6144)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  96)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  97)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              8192)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       8192)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  128)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  129)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              10240)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       10240)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  160)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  161)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              12288)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       12288)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  192)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  193)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              14336)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       14336)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  224)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  225)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              16384)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       16384)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  256)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  257)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              18432)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       18432)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  288)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  289)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              20480)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       20480)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  320)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  321)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              22528)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       22528)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  352)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  353)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              24576)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       24576)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  384)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  385)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              26624)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       26624)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  416)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  417)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              28672)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       28672)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  448)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  449)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              30720)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       30720)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  480)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  481)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              32768)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       32768)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  512)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  513)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              34816)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       34816)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  544)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  545)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              36864)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       36864)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  576)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  577)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              38912)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       38912)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  608)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  609)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              40960)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       40960)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  640)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  641)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              43008)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       43008)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  672)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  673)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              45056)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       45056)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  704)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  705)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                       (((int)threadIdx.x) * 2))] =
          relu(input0[(
              ((((((((int)threadIdx.z) * 128) +
                   ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >>
                     2) *
                    64)) +
                  (((int)blockIdx.y) * 32)) +
                 ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) *
                  8)) +
                (((int)blockIdx.x) * 4)) +
               ((((int)threadIdx.x) & 1) * 2)) +
              47104)]);
      pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                        (((int)threadIdx.x) * 2)) +
                       1)] =
          relu(input0[(((((((((int)threadIdx.z) * 128) +
                            ((((((int)threadIdx.y) * 2) +
                               (((((int)threadIdx.x) * 2) + 1) >> 2)) >>
                              2) *
                             64)) +
                           (((int)blockIdx.y) * 32)) +
                          ((((((int)threadIdx.y) * 2) +
                             (((((int)threadIdx.x) * 2) + 1) >> 2)) &
                            3) *
                           8)) +
                         (((int)blockIdx.x) * 4)) +
                        (((((int)threadIdx.x) * 2) + 1) & 3)) +
                       47104)]);
      input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                     (((int)threadIdx.x) * 2))] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  736)];
      input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) +
                      (((int)threadIdx.x) * 2)) +
                     1)] =
          input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) +
                    (((int)threadIdx.y) * 8)) +
                   (((int)threadIdx.x) * 2)) +
                  737)];
      __syncthreads();
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] *
            input1_shared[(((int)threadIdx.z) * 32)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             16)] *
            input1_shared[((((int)threadIdx.z) * 32) + 1)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             32)] *
            input1_shared[((((int)threadIdx.z) * 32) + 2)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             48)] *
            input1_shared[((((int)threadIdx.z) * 32) + 3)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             64)] *
            input1_shared[((((int)threadIdx.z) * 32) + 4)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             80)] *
            input1_shared[((((int)threadIdx.z) * 32) + 5)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             96)] *
            input1_shared[((((int)threadIdx.z) * 32) + 6)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             112)] *
            input1_shared[((((int)threadIdx.z) * 32) + 7)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             128)] *
            input1_shared[((((int)threadIdx.z) * 32) + 8)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             144)] *
            input1_shared[((((int)threadIdx.z) * 32) + 9)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             160)] *
            input1_shared[((((int)threadIdx.z) * 32) + 10)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             176)] *
            input1_shared[((((int)threadIdx.z) * 32) + 11)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             192)] *
            input1_shared[((((int)threadIdx.z) * 32) + 12)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             208)] *
            input1_shared[((((int)threadIdx.z) * 32) + 13)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             224)] *
            input1_shared[((((int)threadIdx.z) * 32) + 14)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             240)] *
            input1_shared[((((int)threadIdx.z) * 32) + 15)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             256)] *
            input1_shared[((((int)threadIdx.z) * 32) + 16)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             272)] *
            input1_shared[((((int)threadIdx.z) * 32) + 17)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             288)] *
            input1_shared[((((int)threadIdx.z) * 32) + 18)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             304)] *
            input1_shared[((((int)threadIdx.z) * 32) + 19)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             320)] *
            input1_shared[((((int)threadIdx.z) * 32) + 20)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             336)] *
            input1_shared[((((int)threadIdx.z) * 32) + 21)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             352)] *
            input1_shared[((((int)threadIdx.z) * 32) + 22)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             368)] *
            input1_shared[((((int)threadIdx.z) * 32) + 23)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             384)] *
            input1_shared[((((int)threadIdx.z) * 32) + 24)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             400)] *
            input1_shared[((((int)threadIdx.z) * 32) + 25)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             416)] *
            input1_shared[((((int)threadIdx.z) * 32) + 26)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             432)] *
            input1_shared[((((int)threadIdx.z) * 32) + 27)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             448)] *
            input1_shared[((((int)threadIdx.z) * 32) + 28)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             464)] *
            input1_shared[((((int)threadIdx.z) * 32) + 29)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             480)] *
            input1_shared[((((int)threadIdx.z) * 32) + 30)]));
      compute_local[0] =
          (compute_local[0] +
           (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) +
                             496)] *
            input1_shared[((((int)threadIdx.z) * 32) + 31)]));
      compute[((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) +
                  (((int)blockIdx.y) * 32)) +
                 (((int)threadIdx.y) * 8)) +
                (((int)blockIdx.x) * 4)) +
               ((int)threadIdx.x))] = compute_local[0];
    }
  }
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_148(
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {

  __shared__ char shared_buffer[4096];

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    Convolution_float_float_float_cuda_Convolution_1540_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, shared_buffer);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    Convolution_float_float_float_cuda_Convolution_1540_block_kernel(
        input0, input2, output1, threadIdx.x, blockIdx.x - 32, shared_buffer);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_148_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *output0,
    float *output1) {
  BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_148<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
