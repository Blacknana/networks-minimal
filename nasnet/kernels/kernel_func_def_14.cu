// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
// Node name:	Constant_2986
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2986_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2986(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2986_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2986_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2512
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2512_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2512(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2512_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2512_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_238
// Description:	Constant
// Input:
// Output:
//	- name: Constant_238_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_238(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_238_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_238_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_18
// Description:	Constant
// Input:
// Output:
//	- name: Constant_18_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_18(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_18_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_18_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[1152];
  bin_file.read(tmp_mem, 1152);
  cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2011
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2011_0	type: float	shape: Shape{96, 3, 3, 3}
void Constant_float_cuda_Constant_2011(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2011_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2011_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[10368];
  bin_file.read(tmp_mem, 10368);
  cudaMemcpyAsync(output0, tmp_mem, 10368, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3150
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3150_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3150(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3150_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3150_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2973
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2973_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2973(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2973_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2973_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2710
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2710_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2710(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2710_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2710_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2888
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2888_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2888(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2888_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2888_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2518
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2518_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2518(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2518_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2518_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2371
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2371_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2371(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2371_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2371_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2788
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2788_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2788(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2788_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2788_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_188
// Description:	Constant
// Input:
// Output:
//	- name: Constant_188_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_188(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_188_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_188_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[4608];
  bin_file.read(tmp_mem, 4608);
  cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3038
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3038_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3038(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3038_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3038_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2916_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1264_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Slice_1220_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2917_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1266_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: BatchNormInference_1202_0	type: float	shape: Shape{1,
//64, 16, 16}
// Output:
//	- name: Add_1273_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_1274_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_46<<<dim3(32, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_1264_0, Constant_2916_0, Slice_1220_0,
// Add_1273_0); FusedKernel_float_float_float_float_cuda_Add_Add_47<<<dim3(32,
// 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1266_0, Constant_2917_0,
// BatchNormInference_1202_0, Add_1274_0); Deduped function map:
// <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_47 :
// FusedKernel_float_float_float_float_cuda_Add_Add_46

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1264_0	type: float	shape: Shape{1, 64, 16,
//16}
//	- name: Constant_2916_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Slice_1220_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1273_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2469<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_1264_0, Constant_2916_0, BatchNormInference_1270_0);
// Add_float_float_float_cuda_Add_1273<<<dim3(32, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_1270_0, Slice_1220_0, Add_1273_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Add_46_block_kernel(
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
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_109(
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31) {
    FusedKernel_float_float_float_float_cuda_Add_Add_46_block_kernel(
        input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63) {
    FusedKernel_float_float_float_float_cuda_Add_Add_46_block_kernel(
        input4, input3, input5, output1, threadIdx.x, blockIdx.x - 32, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_109_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *input4,
    float *input5, float *output0, float *output1) {
  BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_109<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4,
                                    input5, output0, output1);
}
