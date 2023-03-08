#include "cuda.h"
#include "cuda_runtime.h"
#include <absl/types/span.h>
#include <algorithm>
#include <chrono>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <mma.h>
#include <vector>

template <class T>
static CUresult LaunchKernel(CUfunction f, unsigned grid_x, unsigned block_x,
                             CUstream stream, const T &param,
                             unsigned shared = 0) {
  size_t size = sizeof(T);
  void *config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, const_cast<T *>(&param),
                    CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
  return cuLaunchKernel(f, grid_x, 1, 1, block_x, 1, 1, shared, stream, nullptr,
                        config);
}

#define CU_CHECK(error)                                                        \
  {                                                                            \
    if (error != CUDA_SUCCESS) {                                               \
      const char *error_name;                                                  \
      cuGetErrorName(error, &error_name);                                      \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error, error_name);            \
      exit(1);                                                                 \
    }                                                                          \
  }

#define CUDA_CHECK(error)                                                      \
  {                                                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

namespace lstm {
enum LSTMScaleParams {
  kLstmGateNumber = 4,
  kHiddenSize = 256,
  kInputSize = 256,
  kCellNumber = 10,
  kLstmTimestep = 100,
};

enum LSTMKernelScaleParams {
  kThreadsPerWarp = 32,
  kWarpsPerBlock = 8,
  kColumnsPerBlock = kThreadsPerWarp,
  kGemvBlockNumber = kHiddenSize / kColumnsPerBlock,
  kRowsPerWarp = kHiddenSize / kWarpsPerBlock,
};

struct CellModel {
  float weights_w[kLstmGateNumber][kInputSize][kHiddenSize];
  float weights_u[kLstmGateNumber][kHiddenSize][kHiddenSize];
  float bias[kLstmGateNumber][kHiddenSize];
};
static_assert(sizeof(CellModel) == sizeof(CellModel::weights_w) +
                                       sizeof(CellModel::weights_u) +
                                       sizeof(CellModel::bias),
              "Expect the data to be placed continuously.");

#pragma pack(push, 1)
struct ModelParams {
  CellModel cell_model[kCellNumber];
};
#pragma pack(pop)

struct CellState {
  float data[kHiddenSize];
};

struct CellTemp {
  float data[kLstmGateNumber][kHiddenSize];
};

#pragma pack(push, 1)
struct CellParams {
  CellState cell_state_h[kCellNumber + 1][kLstmTimestep + 1];
  CellState cell_state_c[kCellNumber];
  CellTemp cell_temp[kCellNumber];
};
#pragma pack(pop)
} // namespace lstm

using namespace lstm;

#pragma pack(push, 1)
struct WaveKernelParams {
  CUdeviceptr d_model_params;
  CUdeviceptr d_cell_params;
  int step_start_num;
  int layer_start_num;
};
#pragma pack(pop)

class Wave {
public:
  explicit Wave();
  void InitCellParams(CUdeviceptr d_cell_params);
  void Compute(int wave_size, WaveKernelParams kernel_params);
  void Finalize();

private:
  CUdevice cu_device_;
  CUcontext cu_context_;
  CUfunction cu_wave_compute_;
};

class WavefrontLSTM {
public:
  explicit WavefrontLSTM(absl::Span<const float> src_model);
  bool Initialize(absl::Span<const float> input);
  void Solve();
  bool Fetch(absl::Span<float> output);
  void Finalize();

private:
  Wave wave_;
  CUdeviceptr d_model_params_;
  CUdeviceptr d_cell_params_;
  CUdeviceptr d_input_;
  CUdeviceptr d_output_;
};

__global__ void wave_compute(ModelParams *d_model_params,
                             CellParams *d_cell_params, int step_start_num,
                             int layer_start_num);

int main() {
  std::vector<float> input(sizeof(CellState) * kLstmTimestep / sizeof(float));
  std::vector<float> model(sizeof(ModelParams));
  std::vector<float> output_buffer(sizeof(CellState) * kLstmTimestep /
                                   sizeof(float));
  absl::Span<float> output(output_buffer);
  auto network = new WavefrontLSTM(model);

  enum { kWarmUp = 200, kLoop = 1000 };
  // Warm-up, 100 times
  for (int i = 0; i < kWarmUp; i++) {
    network->Initialize(input);
    network->Solve();
    network->Fetch(output);
  }
  double min_ms = std::numeric_limits<double>::max();
  double max_ms = std::numeric_limits<double>::min();
  double total_ms = 0.00000f;
  for (int i = 0; i < kLoop; i++) {
    auto start = std::chrono::steady_clock::now();
    network->Initialize(input);
    network->Solve();
    network->Fetch(output);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;
    double iteration_ms = elapsed.count();
    printf("Iteration time %f us\n", iteration_ms);
    min_ms = std::min(iteration_ms, min_ms);
    max_ms = std::max(iteration_ms, max_ms);
    total_ms = total_ms + iteration_ms;
  }
  printf("Sumamry: [min, max, mean] = [%f, %f, %f] us\n", min_ms, max_ms,
         total_ms / kLoop);

  network->Finalize();

  return 0;
}

Wave::Wave() {
  CU_CHECK(cuInit(0));
  CU_CHECK(cuDeviceGet(&cu_device_, 0));
  CU_CHECK(cuCtxCreate(&cu_context_, 0, cu_device_));
  CUDA_CHECK(
      cudaGetFuncBySymbol(&cu_wave_compute_, (const void *)wave_compute));
}

void Wave::Finalize() { CU_CHECK(cuCtxDestroy(cu_context_)); }

void Wave::Compute(int wave_size, WaveKernelParams kernel_params) {
  CU_CHECK(LaunchKernel(cu_wave_compute_, kGemvBlockNumber * wave_size,
                        kHiddenSize, 0, kernel_params));
}

void Wave::InitCellParams(CUdeviceptr d_cell_params) {
  CU_CHECK(cuMemsetD32(
      d_cell_params, 0.000000e+00f,
      (sizeof(CellParams::cell_state_h) + sizeof(CellParams::cell_state_c)) /
          sizeof(float)));
}

WavefrontLSTM::WavefrontLSTM(absl::Span<const float> src_model) {
  CU_CHECK(cuMemAlloc(&d_model_params_, sizeof(ModelParams)));
  CU_CHECK(cuMemAlloc(&d_cell_params_, sizeof(CellParams)));
  CU_CHECK(
      cuMemcpyHtoD(d_model_params_, src_model.data(), sizeof(ModelParams)));

  d_input_ = d_cell_params_ + sizeof(CellState);
  d_output_ = d_cell_params_ + sizeof(CellParams::cell_state_h) -
              kLstmTimestep * sizeof(CellState);
}

bool WavefrontLSTM::Initialize(absl::Span<const float> input) {
  if (input.size() != sizeof(CellState) * kLstmTimestep / sizeof(float)) {
    return false;
  }

  wave_.InitCellParams(d_cell_params_);
  CU_CHECK(
      cuMemcpyHtoD(d_input_, input.data(), sizeof(CellState) * kLstmTimestep));
  return true;
}

void WavefrontLSTM::Solve() {
  const int max_wave_size = std::min(kCellNumber, kLstmTimestep);
  const int max_wave_number = kCellNumber + kLstmTimestep - 1;

  for (int wave_idx = 1; wave_idx <= max_wave_number; ++wave_idx) {
    int wave_size = (wave_idx < std::max(kCellNumber, kLstmTimestep))
                        ? std::min(wave_idx, max_wave_size)
                        : (max_wave_size -
                           (wave_idx - std::max(kCellNumber, kLstmTimestep)));
    int step_start_num = (wave_idx < kLstmTimestep) ? wave_idx : kLstmTimestep;
    int layer_start_num =
        (wave_idx < kLstmTimestep) ? 1 : (wave_idx - kLstmTimestep + 1);

    WaveKernelParams kernel_params = {d_model_params_, d_cell_params_,
                                      step_start_num, layer_start_num};
    wave_.Compute(wave_size, kernel_params);
  }
}

bool WavefrontLSTM::Fetch(absl::Span<float> output) {
  if (output.size() != sizeof(CellState) * kLstmTimestep / sizeof(float)) {
    return false;
  }

  CU_CHECK(cuMemcpyDtoH(output.data(), d_output_,
                        sizeof(CellState) * kLstmTimestep));
  return true;
}

void WavefrontLSTM::Finalize() {
  CU_CHECK(cuMemFree(d_model_params_));
  CU_CHECK(cuMemFree(d_cell_params_));
  wave_.Finalize();
}

__device__ static inline float sigmoid(float x) {
  return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}

__global__ void __launch_bounds__(256, 4)
    wave_compute(ModelParams *d_model_params, CellParams *d_cell_params,
                 int step_start_num, int layer_start_num) {
  const int cell_idx = layer_start_num + blockIdx.x / kGemvBlockNumber;
  const int step_idx = step_start_num - blockIdx.x / kGemvBlockNumber;
  CellState *d_input = &d_cell_params->cell_state_h[cell_idx - 1][step_idx];
  CellState *d_input_state_h =
      &d_cell_params->cell_state_h[cell_idx][step_idx - 1];
  CellState *d_output_state_h =
      &d_cell_params->cell_state_h[cell_idx][step_idx];
  CellState *d_state_c = &d_cell_params->cell_state_c[cell_idx - 1];
  CellTemp *d_temp = &d_cell_params->cell_temp[cell_idx - 1];
  CellModel *d_model = &d_model_params->cell_model[cell_idx - 1];

  const int warp_idx = threadIdx.x / kThreadsPerWarp;
  const int lane_idx = threadIdx.x % kThreadsPerWarp;
  const int col_idx =
      (blockIdx.x % kGemvBlockNumber) * kColumnsPerBlock + lane_idx;

  if (warp_idx == 0) {
    for (int i = 0; i < kLstmGateNumber; ++i) {
      d_temp->data[i][col_idx] = 0.000000e+00f;
    }
  }
  __syncthreads();

  float temp[kLstmGateNumber] = {0.000000e+00f, 0.000000e+00f, 0.000000e+00f,
                                 0.000000e+00f};
  const int row_start_idx = kRowsPerWarp * warp_idx;
  const int row_end_idx = row_start_idx + kRowsPerWarp;
  for (int row_idx = row_start_idx; row_idx < row_end_idx; ++row_idx) {
    float input_data = d_input->data[row_idx];
    float state_h_data = d_input_state_h->data[row_idx];
    for (int i = 0; i < kLstmGateNumber; ++i) {
      temp[i] =
          fma(d_model->weights_w[i][row_idx][col_idx], input_data, temp[i]);
    }
    for (int i = 0; i < kLstmGateNumber; ++i) {
      temp[i] =
          fma(d_model->weights_u[i][row_idx][col_idx], state_h_data, temp[i]);
    }
  }

  for (int i = 0; i < kLstmGateNumber; ++i) {
    atomicAdd(&d_temp->data[i][col_idx], temp[i]);
  }
  __syncthreads();

  if (warp_idx == 0) {
    float input_gate_x = d_temp->data[0][col_idx] + d_model->bias[0][col_idx];
    float input_gate_y = d_temp->data[1][col_idx] + d_model->bias[1][col_idx];
    float forget_gate = d_temp->data[2][col_idx] + d_model->bias[2][col_idx];
    float output_gate = d_temp->data[3][col_idx] + d_model->bias[3][col_idx];
    input_gate_x = sigmoid(input_gate_x);
    input_gate_y = tanh(input_gate_y);
    output_gate = sigmoid(output_gate);
    forget_gate =
        sigmoid(forget_gate + 1.000000e+00f) * d_state_c->data[col_idx];
    d_state_c->data[col_idx] = fma(input_gate_x, input_gate_y, forget_gate);
    d_output_state_h->data[col_idx] =
        (tanh(d_state_c->data[col_idx])) * output_gate;
  }
}