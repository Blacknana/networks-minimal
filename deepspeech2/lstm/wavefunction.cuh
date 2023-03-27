#pragma once
#include "lstm_params.h"

#define COLUMNS_PER_BLOCK 32 // one block compute 32 colums
#define THREAD_NUMS_PER_BLOCK 256
#define THREAD_NUMS_PER_BLOCK_1024 1024
#define HIDDENSIZE 256
#define INPUTSIZE HIDDENSIZE
#define INPUTSIZE1 2752

#define compute_cell0_wi(cell, step)                                           \
  {                                                                            \
    onekernel_fuse_opt_v2_2752_256_WI_no_float4_with_wu_global(                \
        blockIdx.x & 0x7, input + step * 7 + cell, model_cell0,                \
        output + step * 7 + cell);                                             \
  }

// 7 is layer num
#define call_onekernel_cell0(cell, step)                                       \
  {                                                                            \
    onekernel_fuse_opt_v2_2752_256_no_float4_with_wu_global(                   \
        blockIdx.x & 0x7, input + step * 7 + cell, model_cell0,                \
        output + step * 7 + cell);                                             \
  }

#define call_onekernel_cell1_6(cell, step)                                     \
  {                                                                            \
    onekernel_fuse_opt_v2_256_256_no_float4_with_wu_global(                    \
        blockIdx.x & 0x7, input + step * 7 + cell, model_cell1_6 + cell - 1,   \
        output + step * 7 + cell);                                             \
  }

#include <stdio.h>
__device__ static inline float sigmoid(float x) {
  return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}

__device__ void onekernel_fuse_opt_v2_2752_256_WI_no_float4_with_wu_global(
    dim3 blockIdx1, WaveInputParams_75_7 *__restrict__ input,
    WaveModelParams_75_7_Cell0 *__restrict__ model,
    WaveOutputParams_75_7 *__restrict__ output) {
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 0x1f;
  const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
  model->temp[0][colOffset] = 0.0;
  model->temp[1][colOffset] = 0.0;
  model->temp[2][colOffset] = 0.0;
  model->temp[3][colOffset] = 0.0;
  float temp1[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
  const int ROWS = INPUTSIZE1 / (THREAD_NUMS_PER_BLOCK_1024 / 32);
  int vectorRow = ROWS * warp_id;
  int kStart =
      vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
  int kEnd = kStart + ROWS * HIDDENSIZE;
  for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
    const float data = input->input_i[vectorRow];
    temp1[0] = fma(model->weight_ws[0][kStart], data, temp1[0]);
    temp1[1] = fma(model->weight_ws[1][kStart], data, temp1[1]);
    temp1[2] = fma(model->weight_ws[2][kStart], data, temp1[2]);
    temp1[3] = fma(model->weight_ws[3][kStart], data, temp1[3]);
  }
  __syncthreads();
  atomicAdd(&model->temp[0][colOffset], temp1[0]);
  atomicAdd(&model->temp[1][colOffset], temp1[1]);
  atomicAdd(&model->temp[2][colOffset], temp1[2]);
  atomicAdd(&model->temp[3][colOffset], temp1[3]);
  __syncthreads();
  if (warp_id == 0) {
    model->wiS[0][colOffset] = model->temp[0][colOffset];
    model->wiS[1][colOffset] = model->temp[1][colOffset];
    model->wiS[2][colOffset] = model->temp[2][colOffset];
    model->wiS[3][colOffset] = model->temp[3][colOffset];
  }
}

//
__device__ void onekernel_fuse_opt_v2_2752_256_no_float4_with_wu_global(
    dim3 blockIdx1, WaveInputParams_75_7 *__restrict__ input,
    WaveModelParams_75_7_Cell0 *__restrict__ model,
    WaveOutputParams_75_7 *__restrict__ output) {
  // just compute u * h

  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 0x1f;
  const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
  model->temp[0][colOffset] = 0.0;
  model->temp[1][colOffset] = 0.0;
  model->temp[2][colOffset] = 0.0;
  model->temp[3][colOffset] = 0.0;
  float temp1[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

  const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
  int vectorRow = ROWS * warp_id;
  int kStart =
      vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
  int kEnd = kStart + ROWS * HIDDENSIZE;
  for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
    const float data2 = input->input_h[vectorRow];
    temp1[0] = fma(model->weight_us[0][kStart], data2, temp1[0]);
    temp1[1] = fma(model->weight_us[1][kStart], data2, temp1[1]);
    temp1[2] = fma(model->weight_us[2][kStart], data2, temp1[2]);
    temp1[3] = fma(model->weight_us[3][kStart], data2, temp1[3]);
  }
  __syncthreads();
  atomicAdd(&model->temp[0][colOffset], temp1[0]);
  atomicAdd(&model->temp[1][colOffset], temp1[1]);
  atomicAdd(&model->temp[2][colOffset], temp1[2]);
  atomicAdd(&model->temp[3][colOffset], temp1[3]);
  __syncthreads();
  if (warp_id == 0) {

    float x, y, z, w;
    x = model->temp[0][colOffset] + model->biass[0][colOffset] +
        model->wiS[0][colOffset];
    y = model->temp[1][colOffset] + model->biass[1][colOffset] +
        model->wiS[1][colOffset];
    z = model->temp[2][colOffset] + model->biass[2][colOffset] +
        model->wiS[2][colOffset];
    w = model->temp[3][colOffset] + model->biass[3][colOffset] +
        model->wiS[3][colOffset];

    x = sigmoid(x);
    y = tanh(y);
    w = sigmoid(w);
    z = sigmoid(z) * output->state_c[colOffset];
    output->state_c[colOffset] = fma(x, y, z);
    output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
  }
}

__device__ void onekernel_fuse_opt_v2_256_256_no_float4_with_wu_global(
    dim3 blockIdx1, WaveInputParams_75_7 *__restrict__ input,
    WaveModelParams_75_7_Cell1_6 *__restrict__ model,
    WaveOutputParams_75_7 *__restrict__ output) {
  if (threadIdx.x >= 256)
    return;

  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 0x1f;
  const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
  model->temp[0][colOffset] = 0.0;
  model->temp[1][colOffset] = 0.0;
  model->temp[2][colOffset] = 0.0;
  model->temp[3][colOffset] = 0.0;
  float temp1[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

  const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
  int vectorRow = ROWS * warp_id;
  int kStart =
      vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
  int kEnd = kStart + ROWS * HIDDENSIZE;
  for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
    const float data = input->input_i[vectorRow];
    const float data2 = input->input_h[vectorRow];
    temp1[0] = fma(model->weight_ws[0][kStart], data, temp1[0]);
    temp1[1] = fma(model->weight_ws[1][kStart], data, temp1[1]);
    temp1[2] = fma(model->weight_ws[2][kStart], data, temp1[2]);
    temp1[3] = fma(model->weight_ws[3][kStart], data, temp1[3]);
    temp1[0] = fma(model->weight_us[0][kStart], data2, temp1[0]);
    temp1[1] = fma(model->weight_us[1][kStart], data2, temp1[1]);
    temp1[2] = fma(model->weight_us[2][kStart], data2, temp1[2]);
    temp1[3] = fma(model->weight_us[3][kStart], data2, temp1[3]);
  }
  __syncthreads();
  atomicAdd(&model->temp[0][colOffset], temp1[0]);
  atomicAdd(&model->temp[1][colOffset], temp1[1]);
  atomicAdd(&model->temp[2][colOffset], temp1[2]);
  atomicAdd(&model->temp[3][colOffset], temp1[3]);
  __syncthreads();
  if (warp_id == 0) {

    float x, y, z, w;
    x = model->temp[0][colOffset] + model->biass[0][colOffset];
    y = model->temp[1][colOffset] + model->biass[1][colOffset];
    z = model->temp[2][colOffset] + model->biass[2][colOffset];
    w = model->temp[3][colOffset] + model->biass[3][colOffset];

    x = sigmoid(x);
    y = tanh(y);
    w = sigmoid(w);
    z = sigmoid(z) * output->state_c[colOffset];
    output->state_c[colOffset] = fma(x, y, z);
    output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
  }
}

__global__ void __launch_bounds__(1024, 1)
    lstm75_7_wave_1(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  int step = blockIdx.x >> 3;
  compute_cell0_wi(0, step)
}

__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave0(WaveInputParams_75_7 *__restrict__ input,
                   WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                   WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                   WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 0);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave1(WaveInputParams_75_7 *__restrict__ input,
                   WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                   WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                   WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 1);
    break;
  case 1:
    call_onekernel_cell1_6(1, 0);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave2(WaveInputParams_75_7 *__restrict__ input,
                   WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                   WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                   WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 2);
    break;
  case 1:
    call_onekernel_cell1_6(1, 1);
    break;
  case 2:
    call_onekernel_cell1_6(2, 0);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave3(WaveInputParams_75_7 *__restrict__ input,
                   WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                   WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                   WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 3);
    break;
  case 1:
    call_onekernel_cell1_6(1, 2);
    break;
  case 2:
    call_onekernel_cell1_6(2, 1);
    break;
  case 3:
    call_onekernel_cell1_6(3, 0);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave4(WaveInputParams_75_7 *__restrict__ input,
                   WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                   WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                   WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 4);
    break;
  case 1:
    call_onekernel_cell1_6(1, 3);
    break;
  case 2:
    call_onekernel_cell1_6(2, 2);
    break;
  case 3:
    call_onekernel_cell1_6(3, 1);
    break;
  case 4:
    call_onekernel_cell1_6(4, 0);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave5(WaveInputParams_75_7 *__restrict__ input,
                   WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                   WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                   WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 5);
    break;
  case 1:
    call_onekernel_cell1_6(1, 4);
    break;
  case 2:
    call_onekernel_cell1_6(2, 3);
    break;
  case 3:
    call_onekernel_cell1_6(3, 2);
    break;
  case 4:
    call_onekernel_cell1_6(4, 1);
    break;
  case 5:
    call_onekernel_cell1_6(5, 0);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave6(WaveInputParams_75_7 *__restrict__ input,
                   WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                   WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                   WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 6);
    break;
  case 1:
    call_onekernel_cell1_6(1, 5);
    break;
  case 2:
    call_onekernel_cell1_6(2, 4);
    break;
  case 3:
    call_onekernel_cell1_6(3, 3);
    break;
  case 4:
    call_onekernel_cell1_6(4, 2);
    break;
  case 5:
    call_onekernel_cell1_6(5, 1);
    break;
  case 6:
    call_onekernel_cell1_6(6, 0);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave7(WaveInputParams_75_7 *__restrict__ input,
                   WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                   WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                   WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 7);
    break;
  case 1:
    call_onekernel_cell1_6(1, 6);
    break;
  case 2:
    call_onekernel_cell1_6(2, 5);
    break;
  case 3:
    call_onekernel_cell1_6(3, 4);
    break;
  case 4:
    call_onekernel_cell1_6(4, 3);
    break;
  case 5:
    call_onekernel_cell1_6(5, 2);
    break;
  case 6:
    call_onekernel_cell1_6(6, 1);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave8(WaveInputParams_75_7 *__restrict__ input,
                   WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                   WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                   WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 8);
    break;
  case 1:
    call_onekernel_cell1_6(1, 7);
    break;
  case 2:
    call_onekernel_cell1_6(2, 6);
    break;
  case 3:
    call_onekernel_cell1_6(3, 5);
    break;
  case 4:
    call_onekernel_cell1_6(4, 4);
    break;
  case 5:
    call_onekernel_cell1_6(5, 3);
    break;
  case 6:
    call_onekernel_cell1_6(6, 2);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave9(WaveInputParams_75_7 *__restrict__ input,
                   WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                   WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                   WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 9);
    break;
  case 1:
    call_onekernel_cell1_6(1, 8);
    break;
  case 2:
    call_onekernel_cell1_6(2, 7);
    break;
  case 3:
    call_onekernel_cell1_6(3, 6);
    break;
  case 4:
    call_onekernel_cell1_6(4, 5);
    break;
  case 5:
    call_onekernel_cell1_6(5, 4);
    break;
  case 6:
    call_onekernel_cell1_6(6, 3);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave10(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 10);
    break;
  case 1:
    call_onekernel_cell1_6(1, 9);
    break;
  case 2:
    call_onekernel_cell1_6(2, 8);
    break;
  case 3:
    call_onekernel_cell1_6(3, 7);
    break;
  case 4:
    call_onekernel_cell1_6(4, 6);
    break;
  case 5:
    call_onekernel_cell1_6(5, 5);
    break;
  case 6:
    call_onekernel_cell1_6(6, 4);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave11(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 11);
    break;
  case 1:
    call_onekernel_cell1_6(1, 10);
    break;
  case 2:
    call_onekernel_cell1_6(2, 9);
    break;
  case 3:
    call_onekernel_cell1_6(3, 8);
    break;
  case 4:
    call_onekernel_cell1_6(4, 7);
    break;
  case 5:
    call_onekernel_cell1_6(5, 6);
    break;
  case 6:
    call_onekernel_cell1_6(6, 5);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave12(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 12);
    break;
  case 1:
    call_onekernel_cell1_6(1, 11);
    break;
  case 2:
    call_onekernel_cell1_6(2, 10);
    break;
  case 3:
    call_onekernel_cell1_6(3, 9);
    break;
  case 4:
    call_onekernel_cell1_6(4, 8);
    break;
  case 5:
    call_onekernel_cell1_6(5, 7);
    break;
  case 6:
    call_onekernel_cell1_6(6, 6);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave13(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 13);
    break;
  case 1:
    call_onekernel_cell1_6(1, 12);
    break;
  case 2:
    call_onekernel_cell1_6(2, 11);
    break;
  case 3:
    call_onekernel_cell1_6(3, 10);
    break;
  case 4:
    call_onekernel_cell1_6(4, 9);
    break;
  case 5:
    call_onekernel_cell1_6(5, 8);
    break;
  case 6:
    call_onekernel_cell1_6(6, 7);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave14(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 14);
    break;
  case 1:
    call_onekernel_cell1_6(1, 13);
    break;
  case 2:
    call_onekernel_cell1_6(2, 12);
    break;
  case 3:
    call_onekernel_cell1_6(3, 11);
    break;
  case 4:
    call_onekernel_cell1_6(4, 10);
    break;
  case 5:
    call_onekernel_cell1_6(5, 9);
    break;
  case 6:
    call_onekernel_cell1_6(6, 8);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave15(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 15);
    break;
  case 1:
    call_onekernel_cell1_6(1, 14);
    break;
  case 2:
    call_onekernel_cell1_6(2, 13);
    break;
  case 3:
    call_onekernel_cell1_6(3, 12);
    break;
  case 4:
    call_onekernel_cell1_6(4, 11);
    break;
  case 5:
    call_onekernel_cell1_6(5, 10);
    break;
  case 6:
    call_onekernel_cell1_6(6, 9);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave16(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 16);
    break;
  case 1:
    call_onekernel_cell1_6(1, 15);
    break;
  case 2:
    call_onekernel_cell1_6(2, 14);
    break;
  case 3:
    call_onekernel_cell1_6(3, 13);
    break;
  case 4:
    call_onekernel_cell1_6(4, 12);
    break;
  case 5:
    call_onekernel_cell1_6(5, 11);
    break;
  case 6:
    call_onekernel_cell1_6(6, 10);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave17(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 17);
    break;
  case 1:
    call_onekernel_cell1_6(1, 16);
    break;
  case 2:
    call_onekernel_cell1_6(2, 15);
    break;
  case 3:
    call_onekernel_cell1_6(3, 14);
    break;
  case 4:
    call_onekernel_cell1_6(4, 13);
    break;
  case 5:
    call_onekernel_cell1_6(5, 12);
    break;
  case 6:
    call_onekernel_cell1_6(6, 11);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave18(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 18);
    break;
  case 1:
    call_onekernel_cell1_6(1, 17);
    break;
  case 2:
    call_onekernel_cell1_6(2, 16);
    break;
  case 3:
    call_onekernel_cell1_6(3, 15);
    break;
  case 4:
    call_onekernel_cell1_6(4, 14);
    break;
  case 5:
    call_onekernel_cell1_6(5, 13);
    break;
  case 6:
    call_onekernel_cell1_6(6, 12);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave19(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 19);
    break;
  case 1:
    call_onekernel_cell1_6(1, 18);
    break;
  case 2:
    call_onekernel_cell1_6(2, 17);
    break;
  case 3:
    call_onekernel_cell1_6(3, 16);
    break;
  case 4:
    call_onekernel_cell1_6(4, 15);
    break;
  case 5:
    call_onekernel_cell1_6(5, 14);
    break;
  case 6:
    call_onekernel_cell1_6(6, 13);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave20(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 20);
    break;
  case 1:
    call_onekernel_cell1_6(1, 19);
    break;
  case 2:
    call_onekernel_cell1_6(2, 18);
    break;
  case 3:
    call_onekernel_cell1_6(3, 17);
    break;
  case 4:
    call_onekernel_cell1_6(4, 16);
    break;
  case 5:
    call_onekernel_cell1_6(5, 15);
    break;
  case 6:
    call_onekernel_cell1_6(6, 14);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave21(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 21);
    break;
  case 1:
    call_onekernel_cell1_6(1, 20);
    break;
  case 2:
    call_onekernel_cell1_6(2, 19);
    break;
  case 3:
    call_onekernel_cell1_6(3, 18);
    break;
  case 4:
    call_onekernel_cell1_6(4, 17);
    break;
  case 5:
    call_onekernel_cell1_6(5, 16);
    break;
  case 6:
    call_onekernel_cell1_6(6, 15);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave22(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 22);
    break;
  case 1:
    call_onekernel_cell1_6(1, 21);
    break;
  case 2:
    call_onekernel_cell1_6(2, 20);
    break;
  case 3:
    call_onekernel_cell1_6(3, 19);
    break;
  case 4:
    call_onekernel_cell1_6(4, 18);
    break;
  case 5:
    call_onekernel_cell1_6(5, 17);
    break;
  case 6:
    call_onekernel_cell1_6(6, 16);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave23(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 23);
    break;
  case 1:
    call_onekernel_cell1_6(1, 22);
    break;
  case 2:
    call_onekernel_cell1_6(2, 21);
    break;
  case 3:
    call_onekernel_cell1_6(3, 20);
    break;
  case 4:
    call_onekernel_cell1_6(4, 19);
    break;
  case 5:
    call_onekernel_cell1_6(5, 18);
    break;
  case 6:
    call_onekernel_cell1_6(6, 17);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave24(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 24);
    break;
  case 1:
    call_onekernel_cell1_6(1, 23);
    break;
  case 2:
    call_onekernel_cell1_6(2, 22);
    break;
  case 3:
    call_onekernel_cell1_6(3, 21);
    break;
  case 4:
    call_onekernel_cell1_6(4, 20);
    break;
  case 5:
    call_onekernel_cell1_6(5, 19);
    break;
  case 6:
    call_onekernel_cell1_6(6, 18);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave25(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 25);
    break;
  case 1:
    call_onekernel_cell1_6(1, 24);
    break;
  case 2:
    call_onekernel_cell1_6(2, 23);
    break;
  case 3:
    call_onekernel_cell1_6(3, 22);
    break;
  case 4:
    call_onekernel_cell1_6(4, 21);
    break;
  case 5:
    call_onekernel_cell1_6(5, 20);
    break;
  case 6:
    call_onekernel_cell1_6(6, 19);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave26(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 26);
    break;
  case 1:
    call_onekernel_cell1_6(1, 25);
    break;
  case 2:
    call_onekernel_cell1_6(2, 24);
    break;
  case 3:
    call_onekernel_cell1_6(3, 23);
    break;
  case 4:
    call_onekernel_cell1_6(4, 22);
    break;
  case 5:
    call_onekernel_cell1_6(5, 21);
    break;
  case 6:
    call_onekernel_cell1_6(6, 20);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave27(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 27);
    break;
  case 1:
    call_onekernel_cell1_6(1, 26);
    break;
  case 2:
    call_onekernel_cell1_6(2, 25);
    break;
  case 3:
    call_onekernel_cell1_6(3, 24);
    break;
  case 4:
    call_onekernel_cell1_6(4, 23);
    break;
  case 5:
    call_onekernel_cell1_6(5, 22);
    break;
  case 6:
    call_onekernel_cell1_6(6, 21);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave28(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 28);
    break;
  case 1:
    call_onekernel_cell1_6(1, 27);
    break;
  case 2:
    call_onekernel_cell1_6(2, 26);
    break;
  case 3:
    call_onekernel_cell1_6(3, 25);
    break;
  case 4:
    call_onekernel_cell1_6(4, 24);
    break;
  case 5:
    call_onekernel_cell1_6(5, 23);
    break;
  case 6:
    call_onekernel_cell1_6(6, 22);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave29(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 29);
    break;
  case 1:
    call_onekernel_cell1_6(1, 28);
    break;
  case 2:
    call_onekernel_cell1_6(2, 27);
    break;
  case 3:
    call_onekernel_cell1_6(3, 26);
    break;
  case 4:
    call_onekernel_cell1_6(4, 25);
    break;
  case 5:
    call_onekernel_cell1_6(5, 24);
    break;
  case 6:
    call_onekernel_cell1_6(6, 23);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave30(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 30);
    break;
  case 1:
    call_onekernel_cell1_6(1, 29);
    break;
  case 2:
    call_onekernel_cell1_6(2, 28);
    break;
  case 3:
    call_onekernel_cell1_6(3, 27);
    break;
  case 4:
    call_onekernel_cell1_6(4, 26);
    break;
  case 5:
    call_onekernel_cell1_6(5, 25);
    break;
  case 6:
    call_onekernel_cell1_6(6, 24);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave31(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 31);
    break;
  case 1:
    call_onekernel_cell1_6(1, 30);
    break;
  case 2:
    call_onekernel_cell1_6(2, 29);
    break;
  case 3:
    call_onekernel_cell1_6(3, 28);
    break;
  case 4:
    call_onekernel_cell1_6(4, 27);
    break;
  case 5:
    call_onekernel_cell1_6(5, 26);
    break;
  case 6:
    call_onekernel_cell1_6(6, 25);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave32(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 32);
    break;
  case 1:
    call_onekernel_cell1_6(1, 31);
    break;
  case 2:
    call_onekernel_cell1_6(2, 30);
    break;
  case 3:
    call_onekernel_cell1_6(3, 29);
    break;
  case 4:
    call_onekernel_cell1_6(4, 28);
    break;
  case 5:
    call_onekernel_cell1_6(5, 27);
    break;
  case 6:
    call_onekernel_cell1_6(6, 26);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave33(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 33);
    break;
  case 1:
    call_onekernel_cell1_6(1, 32);
    break;
  case 2:
    call_onekernel_cell1_6(2, 31);
    break;
  case 3:
    call_onekernel_cell1_6(3, 30);
    break;
  case 4:
    call_onekernel_cell1_6(4, 29);
    break;
  case 5:
    call_onekernel_cell1_6(5, 28);
    break;
  case 6:
    call_onekernel_cell1_6(6, 27);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave34(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 34);
    break;
  case 1:
    call_onekernel_cell1_6(1, 33);
    break;
  case 2:
    call_onekernel_cell1_6(2, 32);
    break;
  case 3:
    call_onekernel_cell1_6(3, 31);
    break;
  case 4:
    call_onekernel_cell1_6(4, 30);
    break;
  case 5:
    call_onekernel_cell1_6(5, 29);
    break;
  case 6:
    call_onekernel_cell1_6(6, 28);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave35(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 35);
    break;
  case 1:
    call_onekernel_cell1_6(1, 34);
    break;
  case 2:
    call_onekernel_cell1_6(2, 33);
    break;
  case 3:
    call_onekernel_cell1_6(3, 32);
    break;
  case 4:
    call_onekernel_cell1_6(4, 31);
    break;
  case 5:
    call_onekernel_cell1_6(5, 30);
    break;
  case 6:
    call_onekernel_cell1_6(6, 29);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave36(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 36);
    break;
  case 1:
    call_onekernel_cell1_6(1, 35);
    break;
  case 2:
    call_onekernel_cell1_6(2, 34);
    break;
  case 3:
    call_onekernel_cell1_6(3, 33);
    break;
  case 4:
    call_onekernel_cell1_6(4, 32);
    break;
  case 5:
    call_onekernel_cell1_6(5, 31);
    break;
  case 6:
    call_onekernel_cell1_6(6, 30);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave37(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 37);
    break;
  case 1:
    call_onekernel_cell1_6(1, 36);
    break;
  case 2:
    call_onekernel_cell1_6(2, 35);
    break;
  case 3:
    call_onekernel_cell1_6(3, 34);
    break;
  case 4:
    call_onekernel_cell1_6(4, 33);
    break;
  case 5:
    call_onekernel_cell1_6(5, 32);
    break;
  case 6:
    call_onekernel_cell1_6(6, 31);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave38(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 38);
    break;
  case 1:
    call_onekernel_cell1_6(1, 37);
    break;
  case 2:
    call_onekernel_cell1_6(2, 36);
    break;
  case 3:
    call_onekernel_cell1_6(3, 35);
    break;
  case 4:
    call_onekernel_cell1_6(4, 34);
    break;
  case 5:
    call_onekernel_cell1_6(5, 33);
    break;
  case 6:
    call_onekernel_cell1_6(6, 32);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave39(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 39);
    break;
  case 1:
    call_onekernel_cell1_6(1, 38);
    break;
  case 2:
    call_onekernel_cell1_6(2, 37);
    break;
  case 3:
    call_onekernel_cell1_6(3, 36);
    break;
  case 4:
    call_onekernel_cell1_6(4, 35);
    break;
  case 5:
    call_onekernel_cell1_6(5, 34);
    break;
  case 6:
    call_onekernel_cell1_6(6, 33);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave40(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 40);
    break;
  case 1:
    call_onekernel_cell1_6(1, 39);
    break;
  case 2:
    call_onekernel_cell1_6(2, 38);
    break;
  case 3:
    call_onekernel_cell1_6(3, 37);
    break;
  case 4:
    call_onekernel_cell1_6(4, 36);
    break;
  case 5:
    call_onekernel_cell1_6(5, 35);
    break;
  case 6:
    call_onekernel_cell1_6(6, 34);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave41(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 41);
    break;
  case 1:
    call_onekernel_cell1_6(1, 40);
    break;
  case 2:
    call_onekernel_cell1_6(2, 39);
    break;
  case 3:
    call_onekernel_cell1_6(3, 38);
    break;
  case 4:
    call_onekernel_cell1_6(4, 37);
    break;
  case 5:
    call_onekernel_cell1_6(5, 36);
    break;
  case 6:
    call_onekernel_cell1_6(6, 35);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave42(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 42);
    break;
  case 1:
    call_onekernel_cell1_6(1, 41);
    break;
  case 2:
    call_onekernel_cell1_6(2, 40);
    break;
  case 3:
    call_onekernel_cell1_6(3, 39);
    break;
  case 4:
    call_onekernel_cell1_6(4, 38);
    break;
  case 5:
    call_onekernel_cell1_6(5, 37);
    break;
  case 6:
    call_onekernel_cell1_6(6, 36);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave43(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 43);
    break;
  case 1:
    call_onekernel_cell1_6(1, 42);
    break;
  case 2:
    call_onekernel_cell1_6(2, 41);
    break;
  case 3:
    call_onekernel_cell1_6(3, 40);
    break;
  case 4:
    call_onekernel_cell1_6(4, 39);
    break;
  case 5:
    call_onekernel_cell1_6(5, 38);
    break;
  case 6:
    call_onekernel_cell1_6(6, 37);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave44(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 44);
    break;
  case 1:
    call_onekernel_cell1_6(1, 43);
    break;
  case 2:
    call_onekernel_cell1_6(2, 42);
    break;
  case 3:
    call_onekernel_cell1_6(3, 41);
    break;
  case 4:
    call_onekernel_cell1_6(4, 40);
    break;
  case 5:
    call_onekernel_cell1_6(5, 39);
    break;
  case 6:
    call_onekernel_cell1_6(6, 38);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave45(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 45);
    break;
  case 1:
    call_onekernel_cell1_6(1, 44);
    break;
  case 2:
    call_onekernel_cell1_6(2, 43);
    break;
  case 3:
    call_onekernel_cell1_6(3, 42);
    break;
  case 4:
    call_onekernel_cell1_6(4, 41);
    break;
  case 5:
    call_onekernel_cell1_6(5, 40);
    break;
  case 6:
    call_onekernel_cell1_6(6, 39);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave46(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 46);
    break;
  case 1:
    call_onekernel_cell1_6(1, 45);
    break;
  case 2:
    call_onekernel_cell1_6(2, 44);
    break;
  case 3:
    call_onekernel_cell1_6(3, 43);
    break;
  case 4:
    call_onekernel_cell1_6(4, 42);
    break;
  case 5:
    call_onekernel_cell1_6(5, 41);
    break;
  case 6:
    call_onekernel_cell1_6(6, 40);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave47(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 47);
    break;
  case 1:
    call_onekernel_cell1_6(1, 46);
    break;
  case 2:
    call_onekernel_cell1_6(2, 45);
    break;
  case 3:
    call_onekernel_cell1_6(3, 44);
    break;
  case 4:
    call_onekernel_cell1_6(4, 43);
    break;
  case 5:
    call_onekernel_cell1_6(5, 42);
    break;
  case 6:
    call_onekernel_cell1_6(6, 41);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave48(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 48);
    break;
  case 1:
    call_onekernel_cell1_6(1, 47);
    break;
  case 2:
    call_onekernel_cell1_6(2, 46);
    break;
  case 3:
    call_onekernel_cell1_6(3, 45);
    break;
  case 4:
    call_onekernel_cell1_6(4, 44);
    break;
  case 5:
    call_onekernel_cell1_6(5, 43);
    break;
  case 6:
    call_onekernel_cell1_6(6, 42);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave49(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 49);
    break;
  case 1:
    call_onekernel_cell1_6(1, 48);
    break;
  case 2:
    call_onekernel_cell1_6(2, 47);
    break;
  case 3:
    call_onekernel_cell1_6(3, 46);
    break;
  case 4:
    call_onekernel_cell1_6(4, 45);
    break;
  case 5:
    call_onekernel_cell1_6(5, 44);
    break;
  case 6:
    call_onekernel_cell1_6(6, 43);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave50(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 50);
    break;
  case 1:
    call_onekernel_cell1_6(1, 49);
    break;
  case 2:
    call_onekernel_cell1_6(2, 48);
    break;
  case 3:
    call_onekernel_cell1_6(3, 47);
    break;
  case 4:
    call_onekernel_cell1_6(4, 46);
    break;
  case 5:
    call_onekernel_cell1_6(5, 45);
    break;
  case 6:
    call_onekernel_cell1_6(6, 44);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave51(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 51);
    break;
  case 1:
    call_onekernel_cell1_6(1, 50);
    break;
  case 2:
    call_onekernel_cell1_6(2, 49);
    break;
  case 3:
    call_onekernel_cell1_6(3, 48);
    break;
  case 4:
    call_onekernel_cell1_6(4, 47);
    break;
  case 5:
    call_onekernel_cell1_6(5, 46);
    break;
  case 6:
    call_onekernel_cell1_6(6, 45);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave52(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 52);
    break;
  case 1:
    call_onekernel_cell1_6(1, 51);
    break;
  case 2:
    call_onekernel_cell1_6(2, 50);
    break;
  case 3:
    call_onekernel_cell1_6(3, 49);
    break;
  case 4:
    call_onekernel_cell1_6(4, 48);
    break;
  case 5:
    call_onekernel_cell1_6(5, 47);
    break;
  case 6:
    call_onekernel_cell1_6(6, 46);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave53(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 53);
    break;
  case 1:
    call_onekernel_cell1_6(1, 52);
    break;
  case 2:
    call_onekernel_cell1_6(2, 51);
    break;
  case 3:
    call_onekernel_cell1_6(3, 50);
    break;
  case 4:
    call_onekernel_cell1_6(4, 49);
    break;
  case 5:
    call_onekernel_cell1_6(5, 48);
    break;
  case 6:
    call_onekernel_cell1_6(6, 47);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave54(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 54);
    break;
  case 1:
    call_onekernel_cell1_6(1, 53);
    break;
  case 2:
    call_onekernel_cell1_6(2, 52);
    break;
  case 3:
    call_onekernel_cell1_6(3, 51);
    break;
  case 4:
    call_onekernel_cell1_6(4, 50);
    break;
  case 5:
    call_onekernel_cell1_6(5, 49);
    break;
  case 6:
    call_onekernel_cell1_6(6, 48);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave55(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 55);
    break;
  case 1:
    call_onekernel_cell1_6(1, 54);
    break;
  case 2:
    call_onekernel_cell1_6(2, 53);
    break;
  case 3:
    call_onekernel_cell1_6(3, 52);
    break;
  case 4:
    call_onekernel_cell1_6(4, 51);
    break;
  case 5:
    call_onekernel_cell1_6(5, 50);
    break;
  case 6:
    call_onekernel_cell1_6(6, 49);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave56(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 56);
    break;
  case 1:
    call_onekernel_cell1_6(1, 55);
    break;
  case 2:
    call_onekernel_cell1_6(2, 54);
    break;
  case 3:
    call_onekernel_cell1_6(3, 53);
    break;
  case 4:
    call_onekernel_cell1_6(4, 52);
    break;
  case 5:
    call_onekernel_cell1_6(5, 51);
    break;
  case 6:
    call_onekernel_cell1_6(6, 50);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave57(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 57);
    break;
  case 1:
    call_onekernel_cell1_6(1, 56);
    break;
  case 2:
    call_onekernel_cell1_6(2, 55);
    break;
  case 3:
    call_onekernel_cell1_6(3, 54);
    break;
  case 4:
    call_onekernel_cell1_6(4, 53);
    break;
  case 5:
    call_onekernel_cell1_6(5, 52);
    break;
  case 6:
    call_onekernel_cell1_6(6, 51);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave58(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 58);
    break;
  case 1:
    call_onekernel_cell1_6(1, 57);
    break;
  case 2:
    call_onekernel_cell1_6(2, 56);
    break;
  case 3:
    call_onekernel_cell1_6(3, 55);
    break;
  case 4:
    call_onekernel_cell1_6(4, 54);
    break;
  case 5:
    call_onekernel_cell1_6(5, 53);
    break;
  case 6:
    call_onekernel_cell1_6(6, 52);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave59(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 59);
    break;
  case 1:
    call_onekernel_cell1_6(1, 58);
    break;
  case 2:
    call_onekernel_cell1_6(2, 57);
    break;
  case 3:
    call_onekernel_cell1_6(3, 56);
    break;
  case 4:
    call_onekernel_cell1_6(4, 55);
    break;
  case 5:
    call_onekernel_cell1_6(5, 54);
    break;
  case 6:
    call_onekernel_cell1_6(6, 53);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave60(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 60);
    break;
  case 1:
    call_onekernel_cell1_6(1, 59);
    break;
  case 2:
    call_onekernel_cell1_6(2, 58);
    break;
  case 3:
    call_onekernel_cell1_6(3, 57);
    break;
  case 4:
    call_onekernel_cell1_6(4, 56);
    break;
  case 5:
    call_onekernel_cell1_6(5, 55);
    break;
  case 6:
    call_onekernel_cell1_6(6, 54);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave61(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 61);
    break;
  case 1:
    call_onekernel_cell1_6(1, 60);
    break;
  case 2:
    call_onekernel_cell1_6(2, 59);
    break;
  case 3:
    call_onekernel_cell1_6(3, 58);
    break;
  case 4:
    call_onekernel_cell1_6(4, 57);
    break;
  case 5:
    call_onekernel_cell1_6(5, 56);
    break;
  case 6:
    call_onekernel_cell1_6(6, 55);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave62(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 62);
    break;
  case 1:
    call_onekernel_cell1_6(1, 61);
    break;
  case 2:
    call_onekernel_cell1_6(2, 60);
    break;
  case 3:
    call_onekernel_cell1_6(3, 59);
    break;
  case 4:
    call_onekernel_cell1_6(4, 58);
    break;
  case 5:
    call_onekernel_cell1_6(5, 57);
    break;
  case 6:
    call_onekernel_cell1_6(6, 56);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave63(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 63);
    break;
  case 1:
    call_onekernel_cell1_6(1, 62);
    break;
  case 2:
    call_onekernel_cell1_6(2, 61);
    break;
  case 3:
    call_onekernel_cell1_6(3, 60);
    break;
  case 4:
    call_onekernel_cell1_6(4, 59);
    break;
  case 5:
    call_onekernel_cell1_6(5, 58);
    break;
  case 6:
    call_onekernel_cell1_6(6, 57);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave64(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 64);
    break;
  case 1:
    call_onekernel_cell1_6(1, 63);
    break;
  case 2:
    call_onekernel_cell1_6(2, 62);
    break;
  case 3:
    call_onekernel_cell1_6(3, 61);
    break;
  case 4:
    call_onekernel_cell1_6(4, 60);
    break;
  case 5:
    call_onekernel_cell1_6(5, 59);
    break;
  case 6:
    call_onekernel_cell1_6(6, 58);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave65(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 65);
    break;
  case 1:
    call_onekernel_cell1_6(1, 64);
    break;
  case 2:
    call_onekernel_cell1_6(2, 63);
    break;
  case 3:
    call_onekernel_cell1_6(3, 62);
    break;
  case 4:
    call_onekernel_cell1_6(4, 61);
    break;
  case 5:
    call_onekernel_cell1_6(5, 60);
    break;
  case 6:
    call_onekernel_cell1_6(6, 59);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave66(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 66);
    break;
  case 1:
    call_onekernel_cell1_6(1, 65);
    break;
  case 2:
    call_onekernel_cell1_6(2, 64);
    break;
  case 3:
    call_onekernel_cell1_6(3, 63);
    break;
  case 4:
    call_onekernel_cell1_6(4, 62);
    break;
  case 5:
    call_onekernel_cell1_6(5, 61);
    break;
  case 6:
    call_onekernel_cell1_6(6, 60);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave67(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 67);
    break;
  case 1:
    call_onekernel_cell1_6(1, 66);
    break;
  case 2:
    call_onekernel_cell1_6(2, 65);
    break;
  case 3:
    call_onekernel_cell1_6(3, 64);
    break;
  case 4:
    call_onekernel_cell1_6(4, 63);
    break;
  case 5:
    call_onekernel_cell1_6(5, 62);
    break;
  case 6:
    call_onekernel_cell1_6(6, 61);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave68(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 68);
    break;
  case 1:
    call_onekernel_cell1_6(1, 67);
    break;
  case 2:
    call_onekernel_cell1_6(2, 66);
    break;
  case 3:
    call_onekernel_cell1_6(3, 65);
    break;
  case 4:
    call_onekernel_cell1_6(4, 64);
    break;
  case 5:
    call_onekernel_cell1_6(5, 63);
    break;
  case 6:
    call_onekernel_cell1_6(6, 62);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave69(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 69);
    break;
  case 1:
    call_onekernel_cell1_6(1, 68);
    break;
  case 2:
    call_onekernel_cell1_6(2, 67);
    break;
  case 3:
    call_onekernel_cell1_6(3, 66);
    break;
  case 4:
    call_onekernel_cell1_6(4, 65);
    break;
  case 5:
    call_onekernel_cell1_6(5, 64);
    break;
  case 6:
    call_onekernel_cell1_6(6, 63);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave70(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 70);
    break;
  case 1:
    call_onekernel_cell1_6(1, 69);
    break;
  case 2:
    call_onekernel_cell1_6(2, 68);
    break;
  case 3:
    call_onekernel_cell1_6(3, 67);
    break;
  case 4:
    call_onekernel_cell1_6(4, 66);
    break;
  case 5:
    call_onekernel_cell1_6(5, 65);
    break;
  case 6:
    call_onekernel_cell1_6(6, 64);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave71(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 71);
    break;
  case 1:
    call_onekernel_cell1_6(1, 70);
    break;
  case 2:
    call_onekernel_cell1_6(2, 69);
    break;
  case 3:
    call_onekernel_cell1_6(3, 68);
    break;
  case 4:
    call_onekernel_cell1_6(4, 67);
    break;
  case 5:
    call_onekernel_cell1_6(5, 66);
    break;
  case 6:
    call_onekernel_cell1_6(6, 65);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave72(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 72);
    break;
  case 1:
    call_onekernel_cell1_6(1, 71);
    break;
  case 2:
    call_onekernel_cell1_6(2, 70);
    break;
  case 3:
    call_onekernel_cell1_6(3, 69);
    break;
  case 4:
    call_onekernel_cell1_6(4, 68);
    break;
  case 5:
    call_onekernel_cell1_6(5, 67);
    break;
  case 6:
    call_onekernel_cell1_6(6, 66);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave73(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 73);
    break;
  case 1:
    call_onekernel_cell1_6(1, 72);
    break;
  case 2:
    call_onekernel_cell1_6(2, 71);
    break;
  case 3:
    call_onekernel_cell1_6(3, 70);
    break;
  case 4:
    call_onekernel_cell1_6(4, 69);
    break;
  case 5:
    call_onekernel_cell1_6(5, 68);
    break;
  case 6:
    call_onekernel_cell1_6(6, 67);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave74(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell0(0, 74);
    break;
  case 1:
    call_onekernel_cell1_6(1, 73);
    break;
  case 2:
    call_onekernel_cell1_6(2, 72);
    break;
  case 3:
    call_onekernel_cell1_6(3, 71);
    break;
  case 4:
    call_onekernel_cell1_6(4, 70);
    break;
  case 5:
    call_onekernel_cell1_6(5, 69);
    break;
  case 6:
    call_onekernel_cell1_6(6, 68);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave75(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell1_6(1, 74);
    break;
  case 1:
    call_onekernel_cell1_6(2, 73);
    break;
  case 2:
    call_onekernel_cell1_6(3, 72);
    break;
  case 3:
    call_onekernel_cell1_6(4, 71);
    break;
  case 4:
    call_onekernel_cell1_6(5, 70);
    break;
  case 5:
    call_onekernel_cell1_6(6, 69);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave76(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell1_6(2, 74);
    break;
  case 1:
    call_onekernel_cell1_6(3, 73);
    break;
  case 2:
    call_onekernel_cell1_6(4, 72);
    break;
  case 3:
    call_onekernel_cell1_6(5, 71);
    break;
  case 4:
    call_onekernel_cell1_6(6, 70);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave77(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell1_6(3, 74);
    break;
  case 1:
    call_onekernel_cell1_6(4, 73);
    break;
  case 2:
    call_onekernel_cell1_6(5, 72);
    break;
  case 3:
    call_onekernel_cell1_6(6, 71);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave78(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell1_6(4, 74);
    break;
  case 1:
    call_onekernel_cell1_6(5, 73);
    break;
  case 2:
    call_onekernel_cell1_6(6, 72);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave79(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell1_6(5, 74);
    break;
  case 1:
    call_onekernel_cell1_6(6, 73);
    break;
  }
}
__global__ void __launch_bounds__(256, 1)
    lstm75_7_wave80(WaveInputParams_75_7 *__restrict__ input,
                    WaveModelParams_75_7_Cell0 *__restrict__ model_cell0,
                    WaveModelParams_75_7_Cell1_6 *__restrict__ model_cell1_6,
                    WaveOutputParams_75_7 *__restrict__ output) {
  switch (blockIdx.x >> 3) {
  case 0:
    call_onekernel_cell1_6(6, 74);
    break;
  }
}
