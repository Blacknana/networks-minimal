// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "kernels/nnfusion_rt.h"
#include <cuda_profiler_api.h>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <math.h>
#include <string.h>
#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    cudaError_t result = (x);                                                  \
    if (result != cudaSuccess) {                                               \
      const char *msg = cudaGetErrorString(result);                            \
      std::stringstream safe_call_ss;                                          \
      safe_call_ss << "\nerror: " #x " failed with error"                      \
                   << "\nfile: " << __FILE__ << "\nline: " << __LINE__         \
                   << "\nmsg: " << msg;                                        \
      throw std::runtime_error(safe_call_ss.str());                            \
    }                                                                          \
  } while (0)

void ReadAll(char *result, const std::string &file) {
  std::ifstream is(file, std::ifstream::binary);
  char buf[4096];
  // ASSERT(is && !is.bad() && !is.eof());
  int cur = 0;

  while (!is.bad() && !is.eof()) {
    is.read(buf, sizeof(buf));
    size_t count = is.gcount();
    if (!count) {
      break;
    }
    // ASSERT(!is.bad() && count % sizeof(float) == 0);
    memcpy(result + cur, buf, count);
    cur += count;
  }
}

int main(void) {

  cuda_init();

  // input argument
  float *Parameter_234_0_host, *Parameter_234_0;
  CUDA_SAFE_CALL(
      cudaMallocHost((void **)&Parameter_234_0_host, sizeof(float) * 3072));
  CUDA_SAFE_CALL(cudaMalloc((void **)&Parameter_234_0, sizeof(float) * 3072));

  // output arguments
  float *Result_1763_0_host, *Result_1763_0;
  CUDA_SAFE_CALL(
      cudaMallocHost((void **)&Result_1763_0_host, sizeof(float) * 10));

  // fill input values
  for (int i = 0; i < 3072; ++i)
    Parameter_234_0_host[i] = 1.0f;

  CUDA_SAFE_CALL(cudaMemcpy(Parameter_234_0, Parameter_234_0_host,
                            sizeof(float) * 3072, cudaMemcpyHostToDevice));

  // warm up for 200 iters:
  for (int i_ = 0; i_ < 200; i_++) {
    CUDA_SAFE_CALL(cudaMemcpy(Parameter_234_0, Parameter_234_0_host,
                              sizeof(float) * 3072, cudaMemcpyHostToDevice));
    kernel_entry(Parameter_234_0, &Result_1763_0);
  }

  // GPU time measurement
  float ms_max = std::numeric_limits<float>::min();
  float ms_min = std::numeric_limits<float>::max();
  float ms_total, ms_i;
  cudaEvent_t start, stop, start_i, stop_i;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&start_i);
  cudaEventCreate(&stop_i);

  // time measurement
  cudaEventRecord(start);

  // kernel call
  int steps = 1000;
  cudaProfilerStart();
  for (int i_ = 0; i_ < steps; i_++) {
    cudaEventRecord(start_i, 0);
    CUDA_SAFE_CALL(cudaMemcpy(Parameter_234_0, Parameter_234_0_host,
                              sizeof(float) * 3072, cudaMemcpyHostToDevice));
    kernel_entry(Parameter_234_0, &Result_1763_0);
    CUDA_SAFE_CALL(cudaMemcpy(Result_1763_0_host, Result_1763_0,
                              sizeof(float) * 10, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop_i, 0);
    cudaEventSynchronize(stop_i);
    cudaEventElapsedTime(&ms_i, start_i, stop_i);
    if (ms_i > ms_max)
      ms_max = ms_i;
    if (ms_i < ms_min)
      ms_min = ms_i;
  }
  cudaProfilerStop();
  // time measurement

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms_total, start, stop);
  printf("Summary: [min, max, mean] = [%f, %f, %f] ms\n", ms_min, ms_max,
         ms_total / steps);

  float *result = (float *)malloc(10 * sizeof(float));
  ReadAll((char *)result, "output");
  for (unsigned i = 0; i < 10; ++i) {
    double diff = fabs(Result_1763_0_host[i] - result[i]) / fabs(result[i]);
    if (diff > 0.0001) {
      std::cout << "output: " << Result_1763_0_host[i]
                << ", expect: " << result[i] << std::endl;
      std::cout << "***********failed************* \n";
    }
  }

  // free context
  CUDA_SAFE_CALL(cudaFree(Parameter_234_0));
  cuda_free();

  cudaFreeHost(Parameter_234_0_host);
  cudaFreeHost(Result_1763_0_host);
  return 0;
}
