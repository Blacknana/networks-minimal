#include "cuda.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include <absl/types/span.h>
#include <fstream>

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
  CellTemp gemvw_temp[kCellNumber];
  CellTemp gemvu_temp[kCellNumber];
};
#pragma pack(pop)
} // namespace lstm

using namespace lstm;

class Wave {
public:
  explicit Wave();
  void InitCellParams(CUdeviceptr d_cell_params);
  void Compute(CUdeviceptr d_model_params, CUdeviceptr d_cell_params,
               int step_idx, int cell_idx, int wave_size);
  void Finalize();

private:
  CUdevice cu_device_;
  CUcontext cu_context_;
  CUfunction cu_solve_;
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

__global__ void gemv(CellState *d_input, CellTemp *d_temp,
                     float weights[kLstmGateNumber][kInputSize][kHiddenSize],
                     const int gate_num);
__global__ void solve(CellState *d_output, CellState *d_state_c,
                      CellTemp *d_gevmw_temp, CellTemp *d_gevmu_temp,
                      CellModel *d_model);

std::vector<float> ReadFloatFromFile(const std::string &path) {
  std::ifstream in_file(path);
  in_file.setf(std::ios::fixed, std::ios::floatfield);
  return std::vector<float>(std::istream_iterator<float>(in_file),
                            std::istream_iterator<float>());
}

TEST(TestLSTM, test_wavefront_lstm) {
  auto model = ReadFloatFromFile("model_params.txt");
  auto input = ReadFloatFromFile("input_params.txt");
  auto expect_result = ReadFloatFromFile("expect_results.txt");
  std::vector<float> output_result_buffer(expect_result.size());
  absl::Span<float> output(output_result_buffer);
  auto network = new WavefrontLSTM(model);

  ASSERT_TRUE(network->Initialize(input));
  network->Solve();
  ASSERT_TRUE(network->Fetch(output));
  for (unsigned int i = 0; i < expect_result.size(); ++i) {
    ASSERT_NEAR(output[i], expect_result[i], 1e-5);
  }

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
  printf("Summary: [min, max, mean] = [%f, %f, %f] ms\n", min_ms / 1000.0,
         max_ms / 1000.0, total_ms / kLoop / 1000.0);

  network->Finalize();
}

Wave::Wave() {
  CU_CHECK(cuInit(0));
  CU_CHECK(cuDeviceGet(&cu_device_, 0));
  CU_CHECK(cuCtxCreate(&cu_context_, 0, cu_device_));
  CUDA_CHECK(cudaGetFuncBySymbol(&cu_solve_, (const void *)solve));
}

void Wave::Finalize() { CU_CHECK(cuCtxDestroy(cu_context_)); }

struct SolveParams {
  CellState *d_model_params;
  CellState *d_cell_params;
  CellTemp *d_model_params1;
  CellTemp *d_cell_params1;
  CellModel *d_model_params2;
};
void Wave::Compute(CUdeviceptr d_model_params, CUdeviceptr d_cell_params,
                   int step_idx, int cell_idx, int wave_size) {
  int wave_idx = 0;
  CellState *d_input_w =
      &((CellParams *)d_cell_params)
           ->cell_state_h[cell_idx + wave_idx - 1][step_idx - wave_idx];
  CellTemp *d_gemvw_temp =
      &((CellParams *)d_cell_params)->gemvw_temp[cell_idx + wave_idx - 1];
  CellModel *d_model =
      &((ModelParams *)d_model_params)->cell_model[cell_idx + wave_idx - 1];

  gemv<<<kGemvBlockNumber * wave_size, kHiddenSize>>>(d_input_w, d_gemvw_temp,
                                                      d_model->weights_w, 0);
  gemv<<<kGemvBlockNumber * wave_size, kHiddenSize>>>(d_input_w, d_gemvw_temp,
                                                      d_model->weights_w, 1);
  gemv<<<kGemvBlockNumber * wave_size, kHiddenSize>>>(d_input_w, d_gemvw_temp,
                                                      d_model->weights_w, 2);
  gemv<<<kGemvBlockNumber * wave_size, kHiddenSize>>>(d_input_w, d_gemvw_temp,
                                                      d_model->weights_w, 3);

  CellState *d_input_u =
      &((CellParams *)d_cell_params)
           ->cell_state_h[cell_idx + wave_idx][step_idx - wave_idx - 1];
  CellTemp *d_gemvu_temp =
      &((CellParams *)d_cell_params)->gemvu_temp[cell_idx + wave_idx - 1];

  gemv<<<kGemvBlockNumber * wave_size, kHiddenSize>>>(d_input_u, d_gemvu_temp,
                                                      d_model->weights_u, 0);
  gemv<<<kGemvBlockNumber * wave_size, kHiddenSize>>>(d_input_u, d_gemvu_temp,
                                                      d_model->weights_u, 1);
  gemv<<<kGemvBlockNumber * wave_size, kHiddenSize>>>(d_input_u, d_gemvu_temp,
                                                      d_model->weights_u, 2);
  gemv<<<kGemvBlockNumber * wave_size, kHiddenSize>>>(d_input_u, d_gemvu_temp,
                                                      d_model->weights_u, 3);

  CellState *d_output =
      &((CellParams *)d_cell_params)
           ->cell_state_h[cell_idx + wave_idx][step_idx - wave_idx];
  CellState *d_state_c =
      &((CellParams *)d_cell_params)->cell_state_c[cell_idx + wave_idx - 1];
  solve<<<kGemvBlockNumber * wave_size, kHiddenSize>>>(
      d_output, d_state_c, d_gemvw_temp, d_gemvu_temp, d_model);
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
  for (int i = 0; i < kLstmTimestep; ++i) {
    for (int j = 0; j < kCellNumber; ++j) {
      wave_.Compute(d_model_params_, d_cell_params_, i + 1, j + 1, 1);
    }
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

__global__ void gemv(CellState *d_input1, CellTemp *d_temp1,
                     float weights1[kLstmGateNumber][kInputSize][kHiddenSize],
                     const int gate_num) {

  const int warp_idx = threadIdx.x / kThreadsPerWarp;
  const int lane_idx = threadIdx.x % kThreadsPerWarp;
  const int col_idx =
      (blockIdx.x % kGemvBlockNumber) * kColumnsPerBlock + lane_idx;

  const int wave_idx = blockIdx.x / kGemvBlockNumber;
  CellState *d_input = d_input1 + wave_idx * (kLstmTimestep + 1) - wave_idx;
  CellTemp *d_temp = d_temp1 + wave_idx;

  float *weights2 =
      (float *)weights1 + wave_idx * sizeof(CellModel) / sizeof(float);
  float(*weights)[kInputSize][kHiddenSize] =
      (float(*)[kInputSize][kHiddenSize])(weights2);

  if (warp_idx == 0) {
    d_temp->data[gate_num][col_idx] = 0.000000e+00f;
  }
  __syncthreads();

  float temp = 0.000000e+00f;
  const int row_start_idx = kRowsPerWarp * warp_idx;
  const int row_end_idx = row_start_idx + kRowsPerWarp;
  for (int row_idx = row_start_idx; row_idx < row_end_idx; ++row_idx) {
    float input_data = d_input->data[row_idx];
    temp = fma(weights[gate_num][row_idx][col_idx], input_data, temp);
  }

  atomicAdd(&d_temp->data[gate_num][col_idx], temp);
}

__global__ void solve(CellState *d_output1, CellState *d_state_c1,
                      CellTemp *d_gevmw_temp1, CellTemp *d_gevmu_temp1,
                      CellModel *d_model1) {
  const int warp_idx = threadIdx.x / kThreadsPerWarp;
  const int lane_idx = threadIdx.x % kThreadsPerWarp;
  const int col_idx =
      (blockIdx.x % kGemvBlockNumber) * kColumnsPerBlock + lane_idx;

  const int wave_idx = blockIdx.x / kGemvBlockNumber;
  CellState *d_output = d_output1 + wave_idx * (kLstmTimestep + 1) - wave_idx;
  CellState *d_state_c = d_state_c1 + wave_idx;
  CellTemp *d_gevmw_temp = d_gevmw_temp1 + wave_idx;
  CellTemp *d_gevmu_temp = d_gevmu_temp1 + wave_idx;
  CellModel *d_model = d_model1 + wave_idx;

  if (warp_idx == 0) {
    float input_gate_x = d_gevmw_temp->data[0][col_idx] +
                         d_gevmu_temp->data[0][col_idx] +
                         d_model->bias[0][col_idx];
    float input_gate_y = d_gevmw_temp->data[1][col_idx] +
                         d_gevmu_temp->data[1][col_idx] +
                         d_model->bias[1][col_idx];
    float forget_gate = d_gevmw_temp->data[2][col_idx] +
                        d_gevmu_temp->data[2][col_idx] +
                        d_model->bias[2][col_idx];
    float output_gate = d_gevmw_temp->data[3][col_idx] +
                        d_gevmu_temp->data[3][col_idx] +
                        d_model->bias[3][col_idx];
    input_gate_x = sigmoid(input_gate_x);
    input_gate_y = tanh(input_gate_y);
    output_gate = sigmoid(output_gate);
    forget_gate =
        sigmoid(forget_gate + 1.000000e+00f) * d_state_c->data[col_idx];
    d_state_c->data[col_idx] = fma(input_gate_x, input_gate_y, forget_gate);
    d_output->data[col_idx] = (tanh(d_state_c->data[col_idx])) * output_gate;
  }
}
