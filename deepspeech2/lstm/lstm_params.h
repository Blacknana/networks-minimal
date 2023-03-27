#pragma once
#define INPUT_SIZE_2752 2752
#define HIDDEN_SIZE_256 256
#include "cuda_runtime.h"

struct LSTMHostCellParams_75_7 {
  float *input_dev;
  const float *init_state_c;
  const float *init_state_h;
  const float *W;
  const float *U;
  const float *bias;
};

struct WaveInputParams_75_7 {
  float *input_i;
  float *input_h;
};

struct WaveModelParams_75_7_Cell0 {
  float4 weight_w[INPUT_SIZE_2752 * HIDDEN_SIZE_256];
  float4 weight_u[HIDDEN_SIZE_256 * HIDDEN_SIZE_256];
  float4 bias[HIDDEN_SIZE_256];
  float4 wi[HIDDEN_SIZE_256];

  float weight_ws[4][INPUT_SIZE_2752 * HIDDEN_SIZE_256];
  float weight_us[4][HIDDEN_SIZE_256 * HIDDEN_SIZE_256];
  float biass[4][HIDDEN_SIZE_256];
  float temp[8][HIDDEN_SIZE_256]; //?
  float wiS[4][HIDDEN_SIZE_256];
};

struct WaveModelParams_75_7_Cell1_6 {
  float4 weight_w[HIDDEN_SIZE_256 * HIDDEN_SIZE_256];
  float4 weight_u[HIDDEN_SIZE_256 * HIDDEN_SIZE_256];
  float4 bias[HIDDEN_SIZE_256];

  float weight_ws[4][HIDDEN_SIZE_256 * HIDDEN_SIZE_256];
  float weight_us[4][HIDDEN_SIZE_256 * HIDDEN_SIZE_256];
  float biass[4][HIDDEN_SIZE_256];
  float temp[8][HIDDEN_SIZE_256]; //?
};

struct WaveOutputParams_75_7 {
  float4 *wi;
  float4 *uh;
  float *state_c;
  float *state_h;
};