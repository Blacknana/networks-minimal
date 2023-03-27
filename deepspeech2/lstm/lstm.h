#pragma once
#include "lstm_params.h"
#include <cstdlib>
#include <iostream>
#include <vector>

namespace mica::experiments::lstm {
class LSTMNet_7_75 {
  enum {
    kNumStep = 75,
    kNumLayer = 7,
    kInputSize = 2752,
    kHiddenSize = 256,
    kBatchSize = 1
  };

private:
  /* data */
  float *inputs_dev;
  float *state_c_s, *state_h_s;
  float4 *weights_w, *weights_u, *bias_s;

  WaveInputParams_75_7 *waveInputParams_dev;
  WaveModelParams_75_7_Cell0 *waveModelParams_cell0_dev;
  WaveModelParams_75_7_Cell1_6 *waveModelParams_cell1_6_dev;
  WaveOutputParams_75_7 *waveOutputParams_dev;
  cudaStream_t stream;
  float *output_host;

public:
  LSTMNet_7_75() {}
  void init(const std::vector<LSTMHostCellParams_75_7> &parmas);
  void compute();
  void finalize();
  float *getOutput() {
    cudaMemcpy(output_host, getLastCellAllStep(), sizeof(float) * kHiddenSize,
               cudaMemcpyDeviceToHost);
    return output_host;
  }
  float *getOutputDev() {
    return (state_h_s + ((kNumStep + 1) * kNumLayer - 1) * kHiddenSize);
  }
  float *getLastCellAllStep() {
    return (state_h_s + ((kNumStep + 1) * (kNumLayer - 1) + 1) * kHiddenSize);
  }
};

} // namespace mica::experiments::lstm