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

namespace seq2seq {

enum Seq2seqScaleParams {
  kEncoderCellNumber = 8,
  kDecoderCellNumber = 4,
  kEncoderTimestep = 100,
  kDecoderTimestep = 30,
  kInputSize = 128,
  kHiddenSize = 128,
  kLstmGateNumber = 4,
};

enum Seq2seqKernelScaleParams {
  kThreadsPerWarp = 32,
  kWarpsPerBlock = 8,
  kColumnsPerBlock = kThreadsPerWarp,
  kGemvBlockNumber = kHiddenSize / kColumnsPerBlock,
  kRowsPerWarp = kHiddenSize / kWarpsPerBlock,
};

#pragma pack(push, 4)
struct CellModel {
  float weights_w[kLstmGateNumber][kInputSize][kHiddenSize];
  float weights_u[kLstmGateNumber][kHiddenSize][kHiddenSize];
  float bias[kLstmGateNumber][kHiddenSize];
};

struct ModelParams {
  CellModel encoder_model[kEncoderCellNumber];
  CellModel decoder_model[kDecoderCellNumber];
};

struct CellState {
  float data[kHiddenSize];
};

struct CellTemp {
  float data[kLstmGateNumber][kHiddenSize];
};

struct CellParams {
  CellState encoder_state_h[kEncoderCellNumber + 1][kEncoderTimestep + 1];
  CellState decoder_state_h[(kDecoderTimestep + 1) * kDecoderCellNumber];
  CellState encoder_state_c[kEncoderCellNumber];
  CellState decoder_state_c[kDecoderCellNumber];
  CellTemp encoder_temp[kEncoderCellNumber];
  CellTemp decoder_temp[kDecoderCellNumber];
};
#pragma pack(pop)
} // namespace seq2seq

using namespace seq2seq;

#pragma pack(push, 1)
struct KernelParams {
  CUdeviceptr d_model_params;
  CUdeviceptr d_cell_params;
  int step_idx;
  int cell_idx;
};

struct KernelParams0 {
  CUdeviceptr d_model_params;
  CUdeviceptr d_cell_params;
};
#pragma pack(pop)

class Seq2seqWave {
public:
  Seq2seqWave();
  void InitCellParams(CUdeviceptr d_cell_params);
  void EncoderCompute(int wave_size, KernelParams kernel_params);
  void DecoderCompute(KernelParams kernel_params);
  void Finalize();

private:
  CUdevice cu_device_;
  CUcontext cu_context_;
  CUfunction cu_wave_encoder_;
  CUfunction cu_decoder_;
  CUfunction cu_decoder_layer0_step0_;
};

class Seq2seq {
public:
  explicit Seq2seq(absl::Span<const float> src_model);
  bool Initialize(absl::Span<const float> input);
  void Solve();
  bool Fetch(absl::Span<float> output);
  void Finalize();

private:
  Seq2seqWave wave_;
  CUdeviceptr d_model_params_;
  CUdeviceptr d_cell_params_;
  CUdeviceptr d_input_;
  CUdeviceptr d_output_;
};

__global__ void seq2seq_wave_encoder(ModelParams *d_model_params,
                                     CellParams *d_cell_params,
                                     int step_start_num, int layer_start_num);
__global__ void seq2seq_decoder(ModelParams *d_model_params,
                                CellParams *d_cell_params, int cell_idx,
                                int step_idx);
__global__ void seq2seq_decoder_layer0_step0(ModelParams *d_model_params,
                                             CellParams *d_cell_params);

int main() {
  std::vector<float> input(sizeof(CellState) * kEncoderTimestep /
                           sizeof(float));
  std::vector<float> model(sizeof(ModelParams));
  std::vector<float> output_buffer(sizeof(CellState) * kEncoderTimestep /
                                   sizeof(float));
  absl::Span<float> output(output_buffer);
  auto network = new Seq2seq(model);

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

Seq2seqWave::Seq2seqWave() {
  CU_CHECK(cuInit(0));
  CU_CHECK(cuDeviceGet(&cu_device_, 0));
  CU_CHECK(cuCtxCreate(&cu_context_, 0, cu_device_));
  CUDA_CHECK(cudaGetFuncBySymbol(&cu_wave_encoder_,
                                 (const void *)seq2seq_wave_encoder));
  CUDA_CHECK(cudaGetFuncBySymbol(&cu_decoder_, (const void *)seq2seq_decoder));
  CUDA_CHECK(cudaGetFuncBySymbol(&cu_decoder_layer0_step0_,
                                 (const void *)seq2seq_decoder_layer0_step0));
}

void Seq2seqWave::Finalize() { CU_CHECK(cuCtxDestroy(cu_context_)); }

void Seq2seqWave::EncoderCompute(int wave_size, KernelParams kernel_params) {
  CU_CHECK(LaunchKernel(cu_wave_encoder_, kGemvBlockNumber * wave_size,
                        kHiddenSize, 0, kernel_params));
}

void Seq2seqWave::DecoderCompute(KernelParams kernel_params) {
  if (kernel_params.step_idx == 0 && kernel_params.cell_idx == 0) {
    KernelParams0 kernel_params0 = {kernel_params.d_model_params,
                                    kernel_params.d_cell_params};
    CU_CHECK(LaunchKernel(cu_decoder_layer0_step0_, kGemvBlockNumber,
                          kHiddenSize, 0, kernel_params0));
  } else {
    CU_CHECK(LaunchKernel(cu_decoder_, kGemvBlockNumber, kHiddenSize, 0,
                          kernel_params));
  }
}

void Seq2seqWave::InitCellParams(CUdeviceptr d_cell_params) {
  CU_CHECK(cuMemsetD32(d_cell_params, 0.000000e+00f,
                       sizeof(CellParams) / sizeof(float)));
}

Seq2seq::Seq2seq(absl::Span<const float> src_model) {
  CU_CHECK(cuMemAlloc(&d_model_params_, sizeof(ModelParams)));
  CU_CHECK(cuMemAlloc(&d_cell_params_, sizeof(CellParams)));
  CU_CHECK(
      cuMemcpyHtoD(d_model_params_, src_model.data(), sizeof(ModelParams)));

  d_input_ = d_cell_params_ + sizeof(CellState);
  d_output_ = d_cell_params_ + sizeof(CellParams::encoder_state_h) +
              sizeof(CellParams::decoder_state_h) -
              kDecoderTimestep * sizeof(CellState);
}

bool Seq2seq::Initialize(absl::Span<const float> input) {
  if (input.size() != sizeof(CellState) * kEncoderTimestep / sizeof(float)) {
    return false;
  }

  wave_.InitCellParams(d_cell_params_);
  CU_CHECK(cuMemcpyHtoD(d_input_, input.data(),
                        sizeof(CellState) * kEncoderTimestep));
  return true;
}

void Seq2seq::Solve() {
  const int max_wave_size = std::min(kEncoderCellNumber, kEncoderTimestep);
  const int max_wave_number = kEncoderCellNumber + kEncoderTimestep - 1;
  for (int wave_idx = 1; wave_idx <= max_wave_number; ++wave_idx) {
    int wave_size =
        (wave_idx < std::max(kEncoderCellNumber, kEncoderTimestep))
            ? std::min(wave_idx, max_wave_size)
            : (max_wave_size -
               (wave_idx - std::max(kEncoderCellNumber, kEncoderTimestep)));
    int step_start_num =
        (wave_idx < kEncoderTimestep) ? wave_idx : kEncoderTimestep;
    int layer_start_num =
        (wave_idx < kEncoderTimestep) ? 1 : (wave_idx - kEncoderTimestep + 1);

    KernelParams kernel_params = {d_model_params_, d_cell_params_,
                                  step_start_num, layer_start_num};
    wave_.EncoderCompute(wave_size, kernel_params);
  }

  for (int step_idx = 1; step_idx <= kDecoderTimestep; ++step_idx) {
    for (int cell_idx = 1; cell_idx <= kDecoderCellNumber; ++cell_idx) {
      KernelParams kernel_params = {d_model_params_, d_cell_params_, step_idx,
                                    cell_idx};
      wave_.DecoderCompute(kernel_params);
    }
  }
}

bool Seq2seq::Fetch(absl::Span<float> output) {
  if (output.size() != sizeof(CellState) * kDecoderTimestep / sizeof(float)) {
    return false;
  }

  CU_CHECK(cuMemcpyDtoH(output.data(), d_output_,
                        sizeof(CellState) * kDecoderTimestep));
  return true;
}

void Seq2seq::Finalize() {
  CU_CHECK(cuMemFree(d_model_params_));
  CU_CHECK(cuMemFree(d_cell_params_));
  wave_.Finalize();
}

__device__ static inline float sigmoid(float x) {
  return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}

__device__ static inline void
seq2seq_compute(CellState *d_input, CellState *d_input_state_h,
                CellState *d_output_state_h, CellState *d_state_c,
                CellTemp *d_temp, CellModel *d_model) {
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

__global__ void __launch_bounds__(128, 4)
    seq2seq_wave_encoder(ModelParams *d_model_params, CellParams *d_cell_params,
                         int step_start_num, int layer_start_num) {
  const int cell_idx = layer_start_num + blockIdx.x / kGemvBlockNumber;
  const int step_idx = step_start_num - blockIdx.x / kGemvBlockNumber;
  CellState *d_input = &d_cell_params->encoder_state_h[cell_idx - 1][step_idx];
  CellState *d_input_state_h =
      &d_cell_params->encoder_state_h[cell_idx][step_idx - 1];
  CellState *d_output_state_h =
      &d_cell_params->encoder_state_h[cell_idx][step_idx];
  CellState *d_state_c = &d_cell_params->encoder_state_c[cell_idx - 1];
  CellTemp *d_temp = &d_cell_params->encoder_temp[cell_idx - 1];
  CellModel *d_model = &d_model_params->encoder_model[cell_idx - 1];
  seq2seq_compute(d_input, d_input_state_h, d_output_state_h, d_state_c, d_temp,
                  d_model);
}

__global__ void __launch_bounds__(128, 4)
    seq2seq_decoder(ModelParams *d_model_params, CellParams *d_cell_params,
                    int step_idx, int cell_idx) {
  CellState *d_input =
      &d_cell_params->decoder_state_h[step_idx * kDecoderCellNumber +
                                      (cell_idx - 1) - 1];
  CellState *d_input_state_h =
      &d_cell_params->decoder_state_h[(step_idx - 1) * kDecoderCellNumber +
                                      cell_idx - 1];
  CellState *d_output_state_h =
      &d_cell_params
           ->decoder_state_h[step_idx * kDecoderCellNumber + cell_idx - 1];
  CellState *d_state_c = &d_cell_params->decoder_state_c[cell_idx - 1];
  CellTemp *d_temp = &d_cell_params->decoder_temp[cell_idx - 1];
  CellModel *d_model = &d_model_params->decoder_model[cell_idx - 1];
  seq2seq_compute(d_input, d_input_state_h, d_output_state_h, d_state_c, d_temp,
                  d_model);
}

__global__ void __launch_bounds__(128, 4)
    seq2seq_decoder_layer0_step0(ModelParams *d_model_params,
                                 CellParams *d_cell_params) {
  CellState *d_input = &d_cell_params->encoder_state_c[kEncoderCellNumber - 1];
  CellState *d_input_state_h = &d_cell_params->decoder_state_h[0];
  CellState *d_output_state_h =
      &d_cell_params->decoder_state_h[kDecoderCellNumber];
  CellState *d_state_c = &d_cell_params->decoder_state_c[0];
  CellTemp *d_temp = &d_cell_params->decoder_temp[0];
  CellModel *d_model = &d_model_params->decoder_model[0];
  seq2seq_compute(d_input, d_input_state_h, d_output_state_h, d_state_c, d_temp,
                  d_model);
}