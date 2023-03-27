#include "lstm.h"
#include "wavefunction.cuh"
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
  void *arg_s[] = {&waveInputParams_dev, &waveModelParams_cell0_dev,
                   &waveModelParams_cell1_6_dev, &waveOutputParams_dev};
  cudaLaunchKernel((void *)lstm75_7_wave_1, dim3(600), dim3(1024),
                   (void **)arg_s, 0, stream);
  cudaLaunchKernel((void *)lstm75_7_wave0, dim3(8), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave1, dim3(16), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave2, dim3(24), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave3, dim3(32), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave4, dim3(40), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave5, dim3(48), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave6, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave7, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave8, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave9, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave10, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave11, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave12, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave13, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave14, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave15, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave16, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave17, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave18, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave19, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave20, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave21, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave22, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave23, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave24, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave25, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave26, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave27, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave28, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave29, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave30, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave31, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave32, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave33, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave34, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave35, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave36, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave37, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave38, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave39, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave40, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave41, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave42, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave43, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave44, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave45, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave46, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave47, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave48, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave49, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave50, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave51, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave52, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave53, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave54, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave55, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave56, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave57, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave58, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave59, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave60, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave61, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave62, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave63, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave64, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave65, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave66, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave67, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave68, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave69, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave70, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave71, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave72, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave73, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave74, dim3(56), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave75, dim3(48), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave76, dim3(40), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave77, dim3(32), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave78, dim3(24), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave79, dim3(16), dim3(256), (void **)arg_s,
                   0, stream);

  cudaLaunchKernel((void *)lstm75_7_wave80, dim3(8), dim3(256), (void **)arg_s,
                   0, stream);
  cudaError_t t = cudaDeviceSynchronize();
}

} // namespace mica::experiments::lstm
