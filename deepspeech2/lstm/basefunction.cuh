#pragma once
#include "lstm_params.h"

#define COLUMNS_PER_BLOCK 32 // one block compute 32 colums
#define THREAD_NUMS_PER_BLOCK 256
#define THREAD_NUMS_PER_BLOCK_1024 1024
#define HIDDENSIZE 256
#define INPUTSIZE HIDDENSIZE
#define INPUTSIZE1 2752

#include <stdio.h>
__device__ static inline float sigmoid(float x) {
  return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}

__global__ void gemvw_2752_256(WaveInputParams_75_7 *__restrict__ input,
                               WaveModelParams_75_7_Cell0 *__restrict__ model,
                               WaveOutputParams_75_7 *__restrict__ output,
                               const int gate_num) {
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 0x1f;
  const int colOffset = blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
  model->wiS[gate_num][colOffset] = 0.0;
  float temp1 = 0.0000f;
  const int ROWS = INPUTSIZE1 / (THREAD_NUMS_PER_BLOCK_1024 / 32);
  int vectorRow = ROWS * warp_id;
  int kStart =
      vectorRow * HIDDENSIZE + blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
  int kEnd = kStart + ROWS * HIDDENSIZE;
  for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
    const float data = input->input_i[vectorRow];
    temp1 = fma(model->weight_ws[gate_num][kStart], data, temp1);
  }
  atomicAdd(&model->wiS[gate_num][colOffset], temp1);
}

__global__ void gemvu_2752_256(WaveInputParams_75_7 *__restrict__ input,
                               WaveModelParams_75_7_Cell0 *__restrict__ model,
                               WaveOutputParams_75_7 *__restrict__ output,
                               const int gate_num) {
  // just compute u * h

  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 0x1f;
  const int colOffset = blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
  model->temp[gate_num][colOffset] = 0.0;
  float temp1 = 0.0000f;

  const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
  int vectorRow = ROWS * warp_id;
  int kStart =
      vectorRow * HIDDENSIZE + blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
  int kEnd = kStart + ROWS * HIDDENSIZE;
  for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
    const float data2 = input->input_h[vectorRow];
    temp1 = fma(model->weight_us[gate_num][kStart], data2, temp1);
  }
  atomicAdd(&model->temp[gate_num][colOffset], temp1);
}

__global__ void solve_2752_256(WaveInputParams_75_7 *__restrict__ input,
                               WaveModelParams_75_7_Cell0 *__restrict__ model,
                               WaveOutputParams_75_7 *__restrict__ output) {
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 0x1f;
  const int colOffset = blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
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

__global__ void gemvw_256_256(WaveInputParams_75_7 *__restrict__ input,
                              WaveModelParams_75_7_Cell1_6 *__restrict__ model,
                              WaveOutputParams_75_7 *__restrict__ output,
                              const int gate_num) {

  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 0x1f;
  const int colOffset = blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
  model->temp[gate_num][colOffset] = 0.0;
  float temp1 = 0.0000f;

  const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
  int vectorRow = ROWS * warp_id;
  int kStart =
      vectorRow * HIDDENSIZE + blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
  int kEnd = kStart + ROWS * HIDDENSIZE;
  for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
    const float data = input->input_i[vectorRow];
    temp1 = fma(model->weight_ws[gate_num][kStart], data, temp1);
  }
  atomicAdd(&model->temp[gate_num][colOffset], temp1);
}

__global__ void gemvu_256_256(WaveInputParams_75_7 *__restrict__ input,
                              WaveModelParams_75_7_Cell1_6 *__restrict__ model,
                              WaveOutputParams_75_7 *__restrict__ output,
                              const int gate_num) {

  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 0x1f;
  const int colOffset = blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
  float temp1 = 0.0000f;

  const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
  int vectorRow = ROWS * warp_id;
  int kStart =
      vectorRow * HIDDENSIZE + blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
  int kEnd = kStart + ROWS * HIDDENSIZE;
  for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
    const float data2 = input->input_h[vectorRow];
    temp1 = fma(model->weight_us[gate_num][kStart], data2, temp1);
  }
  atomicAdd(&model->temp[gate_num][colOffset], temp1);
}

__global__ void solve_256_256(WaveInputParams_75_7 *__restrict__ input,
                              WaveModelParams_75_7_Cell1_6 *__restrict__ model,
                              WaveOutputParams_75_7 *__restrict__ output) {
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 0x1f;
  const int colOffset = blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
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