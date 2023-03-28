#include "basefunction.cuh"
#include "lstm.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
namespace mica::experiments::lstm {
void LSTMNet_7_75::init(const std::vector<LSTMHostCellParams_75_7> &parmas) {
  // malloc
  cudaStreamCreate(&stream);
  cudaMalloc(&inputs_dev,
             sizeof(float) * (kNumStep * kInputSize + kNumLayer * kHiddenSize +
                              (kNumStep + 1) * kNumLayer * kHiddenSize));
  cudaMalloc(&waveInputParams_dev,
             sizeof(WaveInputParams_75_7) * kNumStep * kNumLayer);
  cudaMalloc(&waveModelParams_cell0_dev, sizeof(WaveModelParams_75_7_Cell0));
  cudaMalloc(&waveModelParams_cell1_6_dev,
             sizeof(WaveModelParams_75_7_Cell1_6) * (kNumLayer - 1));
  cudaMalloc(&waveOutputParams_dev,
             sizeof(WaveOutputParams_75_7) * kNumStep * kNumLayer);
  output_host = (float *)malloc(sizeof(float) * kHiddenSize);
  WaveInputParams_75_7 *waveInputParams_host = (WaveInputParams_75_7 *)malloc(
      sizeof(WaveInputParams_75_7) * kNumStep * kNumLayer);
  WaveModelParams_75_7_Cell0 *waveModelParams_cell0_host =
      (WaveModelParams_75_7_Cell0 *)malloc(sizeof(WaveModelParams_75_7_Cell0));
  WaveModelParams_75_7_Cell1_6 *waveModelParams_cell1_6_host =
      (WaveModelParams_75_7_Cell1_6 *)malloc(
          sizeof(WaveModelParams_75_7_Cell1_6) * (kNumLayer - 1));
  WaveOutputParams_75_7 *waveOutputParams_host =
      (WaveOutputParams_75_7 *)malloc(sizeof(WaveOutputParams_75_7) * kNumStep *
                                      kNumLayer);
  // compute offset
  state_c_s = inputs_dev + kNumStep * kInputSize;
  state_h_s = state_c_s + kNumLayer * kHiddenSize;
  // memcpy
  // cudaMemcpy(inputs_dev, parmas[0].input,
  //            sizeof(float) * kNumStep * kInputSize,
  //            cudaMemcpyHostToDevice);
  for (int i = 0; i <= kNumStep; ++i) {
    for (int j = 0; j < kNumLayer; ++j) {
      if (i < kNumStep)
        cudaMemcpy(state_c_s + i * kNumLayer + j, parmas[j].init_state_c,
                   sizeof(float) * kHiddenSize, cudaMemcpyHostToDevice);
      cudaMemcpy(state_h_s + i * kNumLayer + j, parmas[j].init_state_h,
                 sizeof(float) * kHiddenSize, cudaMemcpyHostToDevice);
    }
  }
  for (int i = 0; i < kNumLayer; ++i) {
    if (i == 0) {
      memcpy(waveModelParams_cell0_host->weight_w, parmas[i].W,
             sizeof(float4) * kInputSize * kHiddenSize);
      memcpy(waveModelParams_cell0_host->weight_u, parmas[i].U,
             sizeof(float4) * kHiddenSize * kHiddenSize);
      memcpy(waveModelParams_cell0_host->bias, parmas[i].bias,
             sizeof(float4) * kHiddenSize);
    } else {
      memcpy((waveModelParams_cell1_6_host + i - 1)->weight_w, parmas[i].W,
             sizeof(float4) * kHiddenSize * kHiddenSize);
      memcpy((waveModelParams_cell1_6_host + i - 1)->weight_u, parmas[i].U,
             sizeof(float4) * kHiddenSize * kHiddenSize);
      memcpy((waveModelParams_cell1_6_host + i - 1)->bias, parmas[i].bias,
             sizeof(float4) * kHiddenSize);
    }
  }
  for (int i = 0; i < kNumStep; ++i) {
    for (int j = 0; j < kNumLayer; ++j) {
      if (j == 0) {
        (waveInputParams_host + i * kNumLayer + j)->input_i =
            parmas[0].input_dev + i * kInputSize;
      } else {
        // (waveInputParams_host + i * kNumLayer + j)->input_i =
        //     state_h_s + ((i + 1) * kNumLayer + j - 1) * kHiddenSize;
        (waveInputParams_host + i * kNumLayer + j)->input_i =
            state_h_s + (j - 1) * (kNumStep + 1) * kHiddenSize +
            (i + 1) * kHiddenSize;
      }
      (waveInputParams_host + i * kNumLayer + j)->input_h =
          state_h_s + j * (kNumStep + 1) * kHiddenSize + i * kHiddenSize;
      (waveOutputParams_host + i * kNumLayer + j)->state_c =
          state_c_s + kHiddenSize * j;
      (waveOutputParams_host + i * kNumLayer + j)->state_h =
          state_h_s + j * (kNumStep + 1) * kHiddenSize + (i + 1) * kHiddenSize;
    }
  }

  for (int i = 0; i < 1; ++i) {
    float *hostTempWS =
        (float *)malloc(sizeof(float4) * kInputSize * kHiddenSize);
    float *hostTempUS =
        (float *)malloc(sizeof(float4) * kHiddenSize * kHiddenSize);
    float *hostTempBiasS = (float *)malloc(sizeof(float4) * kHiddenSize);
    for (int m = 0; m < kHiddenSize; ++m)
      for (int n = 0; n < kInputSize; ++n) {
        for (int k = 0; k < 4; ++k) {
          hostTempWS[k * kInputSize * kHiddenSize + n * kHiddenSize + m] =
              parmas[i].W[(n * kHiddenSize + m) * 4 + k];
        }
      }
    // compute u  bias
    for (int m = 0; m < kHiddenSize; m++)
      for (int n = 0; n < kHiddenSize; n++) {
        for (int k = 0; k < 4; k++) {
          hostTempUS[k * 256 * 256 + n * 256 + m] =
              parmas[i].U[(n * kHiddenSize + m) * 4 + k];
          hostTempBiasS[k * 256 + m] = parmas[i].bias[m * 4 + k];
        }
      }

    memcpy((waveModelParams_cell0_host)->weight_ws, hostTempWS,
           sizeof(float4) * kInputSize * kHiddenSize);
    memcpy((waveModelParams_cell0_host)->weight_us, hostTempUS,
           sizeof(float4) * kHiddenSize * kHiddenSize);
    memcpy((waveModelParams_cell0_host)->biass, hostTempBiasS,
           sizeof(float4) * kHiddenSize);

    free(hostTempWS);
    free(hostTempUS);
    free(hostTempBiasS);
  }
  for (int i = 0; i < (kNumLayer - 1); ++i) {

    float *hostTempWS =
        (float *)malloc(sizeof(float4) * kHiddenSize * kHiddenSize);
    float *hostTempUS =
        (float *)malloc(sizeof(float4) * kHiddenSize * kHiddenSize);
    float *hostTempBiasS = (float *)malloc(sizeof(float4) * kHiddenSize);
    for (int m = 0; m < kHiddenSize; m++)
      for (int n = 0; n < kHiddenSize; n++) {
        for (int k = 0; k < 4; k++) {

          hostTempWS[k * 256 * 256 + n * 256 + m] =
              parmas[i + 1].W[(n * kHiddenSize + m) * 4 + k];
          hostTempUS[k * 256 * 256 + n * 256 + m] =
              parmas[i + 1].U[(n * kHiddenSize + m) * 4 + k];
          hostTempBiasS[k * 256 + m] = parmas[i + 1].bias[m * 4 + k];
        }
      }

    memcpy((waveModelParams_cell1_6_host + i)->weight_ws, hostTempWS,
           sizeof(float4) * kHiddenSize * kHiddenSize);
    memcpy((waveModelParams_cell1_6_host + i)->weight_us, hostTempUS,
           sizeof(float4) * kHiddenSize * kHiddenSize);
    memcpy((waveModelParams_cell1_6_host + i)->biass, hostTempBiasS,
           sizeof(float4) * kHiddenSize);

    free(hostTempWS);
    free(hostTempUS);
    free(hostTempBiasS);
  }

  cudaMemcpy(waveInputParams_dev, waveInputParams_host,
             sizeof(WaveInputParams_75_7) * kNumLayer * kNumStep,
             cudaMemcpyHostToDevice);
  cudaMemcpy(waveModelParams_cell0_dev, waveModelParams_cell0_host,
             sizeof(WaveModelParams_75_7_Cell0), cudaMemcpyHostToDevice);
  cudaMemcpy(waveModelParams_cell1_6_dev, waveModelParams_cell1_6_host,
             sizeof(WaveModelParams_75_7_Cell1_6) * (kNumLayer - 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(waveOutputParams_dev, waveOutputParams_host,
             sizeof(WaveOutputParams_75_7) * kNumLayer * kNumStep,
             cudaMemcpyHostToDevice);
  free(waveInputParams_host);
  free(waveModelParams_cell0_host);
  free(waveModelParams_cell1_6_host);
  free(waveOutputParams_host);
}

void LSTMNet_7_75::compute() {
  for (int step_idx = 0; step_idx < 75; ++step_idx) {
    gemvw_2752_256<<<dim3(8), dim3(1024), 0, stream>>>(
        waveInputParams_dev + step_idx * 7, waveModelParams_cell0_dev,
        waveOutputParams_dev + step_idx * 7, 0);
    gemvw_2752_256<<<dim3(8), dim3(1024), 0, stream>>>(
        waveInputParams_dev + step_idx * 7, waveModelParams_cell0_dev,
        waveOutputParams_dev + step_idx * 7, 1);
    gemvw_2752_256<<<dim3(8), dim3(1024), 0, stream>>>(
        waveInputParams_dev + step_idx * 7, waveModelParams_cell0_dev,
        waveOutputParams_dev + step_idx * 7, 2);
    gemvw_2752_256<<<dim3(8), dim3(1024), 0, stream>>>(
        waveInputParams_dev + step_idx * 7, waveModelParams_cell0_dev,
        waveOutputParams_dev + step_idx * 7, 3);
  }

  const int max_wave_size = 7;
  const int max_wave_number = 81;
  for (int wave_idx = 0; wave_idx < max_wave_number; ++wave_idx) {
    int step_start_idx = (wave_idx < 75) ? wave_idx : 74;
    int cell_start_idx = (wave_idx < 75) ? 0 : (wave_idx - 74);
    int wave_size =
        (wave_idx < 75) ? std::min(wave_idx + 1, 7) : (81 - wave_idx);

    for (int kernel_idx = 0; kernel_idx < wave_size; ++kernel_idx) {
      const int cell_idx = cell_start_idx + kernel_idx;
      const int step_idx = step_start_idx - kernel_idx;
      if (cell_idx == 0) {
        gemvu_2752_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + 7 * step_idx, waveModelParams_cell0_dev,
            waveOutputParams_dev + 7 * step_idx, 0);
        gemvu_2752_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + 7 * step_idx, waveModelParams_cell0_dev,
            waveOutputParams_dev + 7 * step_idx, 1);
        gemvu_2752_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + 7 * step_idx, waveModelParams_cell0_dev,
            waveOutputParams_dev + 7 * step_idx, 2);
        gemvu_2752_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + 7 * step_idx, waveModelParams_cell0_dev,
            waveOutputParams_dev + 7 * step_idx, 3);
        solve_2752_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + 7 * step_idx, waveModelParams_cell0_dev,
            waveOutputParams_dev + 7 * step_idx);
      } else {
        gemvw_256_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + cell_idx + 7 * step_idx,
            waveModelParams_cell1_6_dev + cell_idx - 1,
            waveOutputParams_dev + cell_idx + 7 * step_idx, 0);
        gemvw_256_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + cell_idx + 7 * step_idx,
            waveModelParams_cell1_6_dev + cell_idx - 1,
            waveOutputParams_dev + cell_idx + 7 * step_idx, 1);
        gemvw_256_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + cell_idx + 7 * step_idx,
            waveModelParams_cell1_6_dev + cell_idx - 1,
            waveOutputParams_dev + cell_idx + 7 * step_idx, 2);
        gemvw_256_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + cell_idx + 7 * step_idx,
            waveModelParams_cell1_6_dev + cell_idx - 1,
            waveOutputParams_dev + cell_idx + 7 * step_idx, 3);
        gemvu_256_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + cell_idx + 7 * step_idx,
            waveModelParams_cell1_6_dev + cell_idx - 1,
            waveOutputParams_dev + cell_idx + 7 * step_idx, 0);
        gemvu_256_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + cell_idx + 7 * step_idx,
            waveModelParams_cell1_6_dev + cell_idx - 1,
            waveOutputParams_dev + cell_idx + 7 * step_idx, 1);
        gemvu_256_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + cell_idx + 7 * step_idx,
            waveModelParams_cell1_6_dev + cell_idx - 1,
            waveOutputParams_dev + cell_idx + 7 * step_idx, 2);
        gemvu_256_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + cell_idx + 7 * step_idx,
            waveModelParams_cell1_6_dev + cell_idx - 1,
            waveOutputParams_dev + cell_idx + 7 * step_idx, 3);
        solve_256_256<<<dim3(8), dim3(256), 0, stream>>>(
            waveInputParams_dev + cell_idx + 7 * step_idx,
            waveModelParams_cell1_6_dev + cell_idx - 1,
            waveOutputParams_dev + cell_idx + 7 * step_idx);
      }
    }
  }
  cudaError_t t = cudaDeviceSynchronize();
}

} // namespace mica::experiments::lstm
