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

struct StepInput {
  float data[kHiddenSize];
};

struct CellRuntime {
  float state_h[kHiddenSize];
  float state_c[kHiddenSize];
  float gemvw_temp[kLstmGateNumber][kHiddenSize];
  float gemvu_temp[kLstmGateNumber][kHiddenSize];
};
static_assert(sizeof(CellRuntime) == sizeof(CellRuntime::state_h) +
                                         sizeof(CellRuntime::state_c) +
                                         sizeof(CellRuntime::gemvw_temp) +
                                         sizeof(CellRuntime::gemvu_temp),
              "Expect the data to be placed continuously.");

#pragma pack(push, 1)
struct InputParams {
  StepInput step_input[kLstmTimestep];
};

struct CellParams {
  CellRuntime cell_runtime[kCellNumber];
};
#pragma pack(pop)
} // namespace lstm

using namespace lstm;

#pragma pack(push, 1)
struct GemvwParams {
  CUdeviceptr d_input;
  CUdeviceptr d_model;
  CUdeviceptr d_runtime;
  int gate_num;
};

struct GemvuParams {
  CUdeviceptr d_model;
  CUdeviceptr d_runtime;
  int gate_num;
};

struct SolveParams {
  CUdeviceptr d_output;
  CUdeviceptr d_model;
  CUdeviceptr d_runtime;
};
#pragma pack(pop)

class NaiveLSTMCell {
public:
  NaiveLSTMCell();
  void CountDeviceptr(CUdeviceptr d_input_params, CUdeviceptr d_model_params);
  void InitCellParams();
  void Compute(int step_idx, int cell_idx);
  void Finalize();

private:
  CUdevice cu_device_;
  CUcontext cu_context_;
  CUfunction cu_gemvw_;
  CUfunction cu_gemvu_;
  CUfunction cu_solve_;
  CUdeviceptr d_cell_params_;
  CUdeviceptr d_cell_model_[kCellNumber];
  CUdeviceptr d_step_input_[kLstmTimestep];
  CUdeviceptr d_cell_runtime_[kCellNumber];
};

class NaiveLSTM {
public:
  explicit NaiveLSTM(absl::Span<const float> src_model);
  bool Initialize(absl::Span<const float> input);
  void Solve();
  bool Fetch(absl::Span<float> output);
  void Finalize();

private:
  NaiveLSTMCell cell_;
  CUdeviceptr d_model_params_;
  CUdeviceptr d_input_params_;
};

__global__ void gemvw(StepInput *d_input, CellModel *d_model,
                      CellRuntime *d_runtime, int gate_num);
__global__ void gemvu(CellModel *d_model, CellRuntime *d_runtime, int gate_num);
__global__ void solve(StepInput *d_output, CellModel *d_model,
                      CellRuntime *d_runtime);

int main() {
  std::vector<float> input(sizeof(InputParams) / sizeof(float));
  std::vector<float> model(sizeof(ModelParams));
  std::vector<float> output_buffer(sizeof(InputParams) / sizeof(float));
  absl::Span<float> output(output_buffer);
  auto network = new NaiveLSTM(model);

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

NaiveLSTMCell::NaiveLSTMCell() {
  CU_CHECK(cuInit(0));
  CU_CHECK(cuDeviceGet(&cu_device_, 0));
  CU_CHECK(cuCtxCreate(&cu_context_, 0, cu_device_));
  CUDA_CHECK(cudaGetFuncBySymbol(&cu_gemvw_, (const void *)gemvw));
  CUDA_CHECK(cudaGetFuncBySymbol(&cu_gemvu_, (const void *)gemvu));
  CUDA_CHECK(cudaGetFuncBySymbol(&cu_solve_, (const void *)solve));
  CU_CHECK(cuMemAlloc(&d_cell_params_, sizeof(CellParams)));
}

void NaiveLSTMCell::Finalize() {
  CU_CHECK(cuMemFree(d_cell_params_));
  CU_CHECK(cuCtxDestroy(cu_context_));
}

void NaiveLSTMCell::CountDeviceptr(CUdeviceptr d_input_params,
                                   CUdeviceptr d_model_params) {
  d_cell_runtime_[0] = d_cell_params_;
  for (int i = 1; i < kCellNumber; ++i) {
    d_cell_runtime_[i] = d_cell_runtime_[i - 1] + sizeof(CellRuntime);
  }

  d_step_input_[0] = d_input_params;
  for (int i = 1; i < kLstmTimestep; ++i) {
    d_step_input_[i] = d_step_input_[i - 1] + sizeof(StepInput);
  }

  d_cell_model_[0] = d_model_params;
  for (int i = 1; i < kCellNumber; ++i) {
    d_cell_model_[i] = d_cell_model_[i - 1] + sizeof(CellModel);
  }
}

void NaiveLSTMCell::Compute(int step_idx, int cell_idx) {
  for (int i = 0; i < kLstmGateNumber; ++i) {
    GemvwParams gemvw_params = {d_step_input_[step_idx],
                                d_cell_model_[cell_idx],
                                d_cell_runtime_[cell_idx], i};
    CU_CHECK(LaunchKernel(cu_gemvw_, kGemvBlockNumber, kHiddenSize, 0,
                          gemvw_params));
  }

  for (int i = 0; i < kLstmGateNumber; ++i) {
    GemvuParams gemvu_params = {d_cell_model_[cell_idx],
                                d_cell_runtime_[cell_idx], i};
    CU_CHECK(LaunchKernel(cu_gemvu_, kGemvBlockNumber, kHiddenSize, 0,
                          gemvu_params));
  }

  SolveParams solve_params = {d_step_input_[step_idx], d_cell_model_[cell_idx],
                              d_cell_runtime_[cell_idx]};
  CU_CHECK(LaunchKernel(cu_solve_, 1, kHiddenSize, 0, solve_params));
}

void NaiveLSTMCell::InitCellParams() {
  CU_CHECK(cuMemsetD32(d_cell_params_, 0.000000e+00f,
                       sizeof(CellParams) / sizeof(float)));
}

NaiveLSTM::NaiveLSTM(absl::Span<const float> src_model) {
  CU_CHECK(cuMemAlloc(&d_input_params_, sizeof(InputParams)));
  CU_CHECK(cuMemAlloc(&d_model_params_, sizeof(ModelParams)));
  CU_CHECK(
      cuMemcpyHtoD(d_model_params_, src_model.data(), sizeof(ModelParams)));
  cell_.CountDeviceptr(d_input_params_, d_model_params_);
}

bool NaiveLSTM::Initialize(absl::Span<const float> input) {
  if (input.size() != sizeof(InputParams) / sizeof(float)) {
    return false;
  }

  CU_CHECK(cuMemcpyHtoD(d_input_params_, input.data(), sizeof(InputParams)));
  cell_.InitCellParams();
  return true;
}

void NaiveLSTM::Solve() {
  for (int i = 0; i < kLstmTimestep; ++i) {
    for (int j = 0; j < kCellNumber; ++j) {
      cell_.Compute(i, j);
    }
  }
}

bool NaiveLSTM::Fetch(absl::Span<float> output) {
  if (output.size() != sizeof(InputParams) / sizeof(float)) {
    return false;
  }

  CU_CHECK(cuMemcpyDtoH(output.data(), d_input_params_, sizeof(InputParams)));
  return true;
}

void NaiveLSTM::Finalize() {
  CU_CHECK(cuMemFree(d_model_params_));
  CU_CHECK(cuMemFree(d_input_params_));
  cell_.Finalize();
}

__device__ static inline float sigmoid(float x) {
  return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}

__global__ void gemvw(StepInput *d_input, CellModel *d_model,
                      CellRuntime *d_runtime, int gate_num) {
  const int warp_idx = threadIdx.x / kThreadsPerWarp;
  const int lane_idx = threadIdx.x % kThreadsPerWarp;
  const int col_idx = blockIdx.x * kColumnsPerBlock + lane_idx;

  if (warp_idx == 0) {
    d_runtime->gemvw_temp[gate_num][col_idx] = 0.000000e+00f;
  }
  __syncthreads();

  float temp = 0.000000e+00f;
  const int row_start_idx = kRowsPerWarp * warp_idx;
  const int row_end_idx = row_start_idx + kRowsPerWarp;
  for (int row_idx = row_start_idx; row_idx < row_end_idx; ++row_idx) {
    float input_data = d_input->data[row_idx];
    temp =
        fma(d_model->weights_w[gate_num][row_idx][col_idx], input_data, temp);
  }

  atomicAdd(&d_runtime->gemvw_temp[gate_num][col_idx], temp);
}

__global__ void gemvu(CellModel *d_model, CellRuntime *d_runtime,
                      int gate_num) {
  const int warp_idx = threadIdx.x / kThreadsPerWarp;
  const int lane_idx = threadIdx.x % kThreadsPerWarp;
  const int col_idx = blockIdx.x * kColumnsPerBlock + lane_idx;

  if (warp_idx == 0) {
    d_runtime->gemvu_temp[gate_num][col_idx] = 0.000000e+00f;
  }
  __syncthreads();

  float temp = 0.000000e+00f;
  const int row_start_idx = kRowsPerWarp * warp_idx;
  const int row_end_idx = row_start_idx + kRowsPerWarp;
  for (int row_idx = row_start_idx; row_idx < row_end_idx; ++row_idx) {
    float state_h_data = d_runtime->state_h[row_idx];
    temp =
        fma(d_model->weights_u[gate_num][row_idx][col_idx], state_h_data, temp);
  }

  atomicAdd(&d_runtime->gemvu_temp[gate_num][col_idx], temp);
}

__global__ void solve(StepInput *d_output, CellModel *d_model,
                      CellRuntime *d_runtime) {
  const int col_idx = threadIdx.x;

  float input_gate_x = d_runtime->gemvw_temp[0][col_idx] +
                       d_runtime->gemvu_temp[0][col_idx] +
                       d_model->bias[0][col_idx];
  float input_gate_y = d_runtime->gemvw_temp[1][col_idx] +
                       d_runtime->gemvu_temp[1][col_idx] +
                       d_model->bias[1][col_idx];
  float forget_gate = d_runtime->gemvw_temp[2][col_idx] +
                      d_runtime->gemvu_temp[2][col_idx] +
                      d_model->bias[2][col_idx];
  float output_gate = d_runtime->gemvw_temp[3][col_idx] +
                      d_runtime->gemvu_temp[3][col_idx] +
                      d_model->bias[3][col_idx];
  input_gate_x = sigmoid(input_gate_x);
  input_gate_y = tanh(input_gate_y);
  output_gate = sigmoid(output_gate);
  forget_gate =
      sigmoid(forget_gate + 1.000000e+00f) * d_runtime->state_c[col_idx];
  d_runtime->state_c[col_idx] = fma(input_gate_x, input_gate_y, forget_gate);
  d_runtime->state_h[col_idx] =
      (tanh(d_runtime->state_c[col_idx])) * output_gate;
  d_output->data[col_idx] = d_runtime->state_h[col_idx];
}