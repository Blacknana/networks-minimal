// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1) { return x0 + x1; }
__device__ __forceinline__ float relu(float x0) { return fmaxf(0, x0); }
// Node name:	Constant_41
// Description:	Constant
// Input:
// Output:
//	- name: Constant_41_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_41(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_41_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_41_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[6400];
  bin_file.read(tmp_mem, 6400);
  cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_217
// Description:	Constant
// Input:
// Output:
//	- name: Constant_217_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_217(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_217_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_217_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[3200];
  bin_file.read(tmp_mem, 3200);
  cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_455
// Description:	Constant
// Input:
// Output:
//	- name: Constant_455_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_455(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_455_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_455_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2815
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2815_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2815(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2815_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2815_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2644
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2644_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2644(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2644_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2644_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_259
// Description:	Constant
// Input:
// Output:
//	- name: Constant_259_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_259(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_259_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_259_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2958
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2958_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2958(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2958_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2958_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2434
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2434_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2434(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2434_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2434_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2446
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2446_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2446(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2446_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2446_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[16384];
  bin_file.read(tmp_mem, 16384);
  cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_104
// Description:	Constant
// Input:
// Output:
//	- name: Constant_104_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_104(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_104_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_104_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[12800];
  bin_file.read(tmp_mem, 12800);
  cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2847
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2847_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2847(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2847_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2847_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2777
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2777_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2777(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2777_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2777_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[131072];
  bin_file.read(tmp_mem, 131072);
  cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_2858
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2858_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2858(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_2858_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_2858_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[65536];
  bin_file.read(tmp_mem, 65536);
  cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	Constant_3156
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3156_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3156(cudaStream_t stream, float *output0) {
  std::ifstream bin_file("./Constant/Constant_3156_0.bin",
                         std::ios::in | std::ios::binary);
  if (bin_file.fail()) {
    printf("Load Constant_3156_0 failed.\n");
    exit(1);
  }
  char *tmp_mem = new char[32768];
  bin_file.read(tmp_mem, 32768);
  cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
  bin_file.close();
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_577_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Constant_2815_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2814_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_575_0	type: float	shape: Shape{1, 32, 32,
// 32}
// Output:
//	- name: BatchNormInference_579_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: BatchNormInference_578_0	type: float	shape: Shape{1,
// 32, 32, 32}
//	- name: Relu_580_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2082<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_577_0, Constant_2815_0, BatchNormInference_579_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_7<<<dim3(64, 1, 1),
// dim3(512, 1, 1), 0, 0>>>(Convolution_575_0, Constant_2814_0, Relu_580_0,
// BatchNormInference_578_0); Deduped function map: <src_function_name :
// deduped_function_name>

// Node name:	Add_2082
// Description:	Add
// Input:
//	- name: Convolution_577_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Constant_2815_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_579_0	type: float	shape: Shape{1,
// 32, 32, 32}
__device__ __forceinline__ void
Add_float_float_float_cuda_Add_2082_block_kernel(float *input0, float *input1,
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
//	- name: Convolution_575_0	type: float	shape: Shape{1, 32, 32,
// 32}
//	- name: Constant_2814_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_580_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_578_0	type: float	shape: Shape{1,
// 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2085<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(Convolution_575_0, Constant_2814_0, BatchNormInference_578_0);
// Relu_float_float_cuda_Relu_580<<<dim3(64, 1, 1), dim3(512, 1, 1), 0,
// 0>>>(BatchNormInference_578_0, Relu_580_0);
__device__ __forceinline__ void
FusedKernel_float_float_float_float_cuda_Add_Relu_7_block_kernel(
    float *input0, float *input1, float *output0, float *output1, int thread_id,
    int block_id, char *shared_buffer) {
  if (thread_id >= 512) {
    return;
  }
  const dim3 blockDim(512, 1, 1);
  const dim3 gridDim(64, 1, 1);
  const dim3 blockIdx(block_id, 0, 0);
  int tid = blockIdx.x * 512 + threadIdx.x;
  float temp0 = add(input0[tid], input1[tid]);
  float temp1 = relu(temp0);
  output1[tid] = temp0;
  output0[tid] = temp1;
}

extern "C" __global__ void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_10(
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {

  if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63) {
    Add_float_float_float_cuda_Add_2082_block_kernel(
        input0, input1, output0, threadIdx.x, blockIdx.x - 0, NULL);
  } else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127) {
    FusedKernel_float_float_float_float_cuda_Add_Relu_7_block_kernel(
        input3, input2, output2, output1, threadIdx.x, blockIdx.x - 64, NULL);
  }
}

extern void
BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_10_Call(
    const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream,
    float *input0, float *input1, float *input2, float *input3, float *output0,
    float *output1, float *output2) {
  BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_10<<<
      grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0,
                                    output1, output2);
}
