
#include "deepspeech.h"
#include "op/op.h"
#include "op/utils.h"
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

enum {
  kBatch = 1,
  kInputStep = 300,
  kOutputStep = 75,
  kNumClass = 29,
  kHiddenSize = 256,
  kFeatureSize = 2752,
  kLstmLayer = 7,
};

std::vector<float> ReadAll(const std::string &file, const std::string &dir) {
  std::vector<float> result;
  std::ifstream is(dir + '/' + file, std::ifstream::binary);
  char buf[4096];
  ASSERT(is && !is.bad() && !is.eof());

  while (!is.bad() && !is.eof()) {
    is.read(buf, sizeof(buf));
    size_t count = is.gcount();
    if (!count) {
      break;
    }
    ASSERT(!is.bad() && count % sizeof(float) == 0);
    result.insert(result.end(), reinterpret_cast<float *>(buf),
                  reinterpret_cast<float *>(buf + count));
  }
  return result;
}

std::vector<float> RandomData(uint64_t size) {
  std::vector<float> data(size);
  enum {
    kRandomSeed = 0xdeadbeef,
  };
  std::default_random_engine random(kRandomSeed);
  std::uniform_real_distribution<float> dist(0, 0.001);
  for (uint64_t i = 0; i < size; ++i) {
    data[i] = dist(random);
  }
  return data;
}

namespace mica::experiments::lstm {
class DeepSpeech2Test : public ::testing::Test {
protected:
  virtual void SetUp() override {
    const std::string &kDataPath = ".";
    const std::string &kOutputFile = "deepspeech.output";
    const std::string &kParamFile = "deepspeech.weight";
    const std::string &kInputFile = "deepspeech.input";

    gpu_memory_ = nullptr;
    pytorch_input_ = ReadAll(kInputFile, kDataPath);
    pytorch_output_ = ReadAll(kOutputFile, kDataPath);

    auto state_size =
        DeepSpeech2Model<kBatch, kOutputStep, kNumClass>::GetStateSize();
    CUDA_CHECK(cudaMalloc(&gpu_memory_, (state_size + pytorch_input_.size() +
                                         kOutputStep * kHiddenSize) *
                                            sizeof(float)));
    model_.reset(
        new DeepSpeech2Model<kBatch, kOutputStep, kNumClass>(gpu_memory_));

    auto params = model_->GetParameters(); // device

    auto param_ = ReadAll(kParamFile, kDataPath);
    auto param_w = ReadAll("w.data", kDataPath);
    auto param_u = ReadAll("u.data", kDataPath);
    auto param_b = ReadAll("bias", kDataPath);
    auto param_w_fuse = ReadAll("w.data", kDataPath);
    auto param_u_fuse = ReadAll("u.data", kDataPath);
    auto param_b_fuse = ReadAll("bias", kDataPath);

    auto host_param = reinterpret_cast<char *>(param_.data());

    for (auto &param : params) {
      if (param.size()) {
        CUDA_CHECK(cudaMemcpy(const_cast<char *>(param.data()), host_param,
                              param.size(), cudaMemcpyHostToDevice));
        host_param += param.size();
      }
    }

    float *all_zero_state = (float *)malloc(sizeof(float) * kHiddenSize);
    for (int i = 0; i < kHiddenSize; ++i)
      all_zero_state[i] = 0.000f;
    auto &lstms = model_->GetLstm();
    float *w_ptr = reinterpret_cast<float *>(param_w.data()),
          *u_ptr = reinterpret_cast<float *>(param_u.data()),
          *b_ptr = reinterpret_cast<float *>(param_b.data());

    float *w_ptr_fuse = reinterpret_cast<float *>(param_w_fuse.data()),
          *u_ptr_fuse = reinterpret_cast<float *>(param_u_fuse.data()),
          *b_ptr_fuse = reinterpret_cast<float *>(param_b_fuse.data());
    std::vector<LSTMHostCellParams_75_7> lstmInitParams;
    float *lstmInputDev = model_->GetPermuteOutput();
    // deal with mem layout

    for (int i = 0; i < kLstmLayer; ++i) {

      if (i > 0) {
        if (i == 1) {
          w_ptr = w_ptr + kFeatureSize * kHiddenSize * 4;
          w_ptr_fuse = w_ptr_fuse + kFeatureSize * kHiddenSize * 4;
        } else {
          w_ptr = w_ptr + kHiddenSize * kHiddenSize * 4;
          w_ptr_fuse = w_ptr_fuse + kHiddenSize * kHiddenSize * 4;
        }
        u_ptr = u_ptr + kHiddenSize * kHiddenSize * 4;
        b_ptr = b_ptr + kHiddenSize * 4;
        u_ptr_fuse = u_ptr_fuse + kHiddenSize * kHiddenSize * 4;
        b_ptr_fuse = b_ptr_fuse + kHiddenSize * 4;
      }

      if (i == 0) {
        for (int j = 0; j < kFeatureSize * kHiddenSize; ++j) {
          w_ptr[j * 4] = w_ptr_fuse[j];
          w_ptr[j * 4 + 1] = w_ptr_fuse[j + kHiddenSize * kFeatureSize];
          w_ptr[j * 4 + 2] = w_ptr_fuse[j + kHiddenSize * kFeatureSize * 2];
          w_ptr[j * 4 + 3] = w_ptr_fuse[j + kHiddenSize * kFeatureSize * 3];
        }

      } else {
        for (int j = 0; j < kHiddenSize * kHiddenSize; ++j) {
          w_ptr[j * 4] = w_ptr_fuse[j];
          w_ptr[j * 4 + 1] = w_ptr_fuse[j + kHiddenSize * kHiddenSize];
          w_ptr[j * 4 + 2] = w_ptr_fuse[j + kHiddenSize * kHiddenSize * 2];
          w_ptr[j * 4 + 3] = w_ptr_fuse[j + kHiddenSize * kHiddenSize * 3];
        }
      }

      for (int j = 0; j < kHiddenSize * kHiddenSize; ++j) {
        u_ptr[j * 4] = u_ptr_fuse[j];
        u_ptr[j * 4 + 1] = u_ptr_fuse[j + kHiddenSize * kHiddenSize];
        u_ptr[j * 4 + 2] = u_ptr_fuse[j + kHiddenSize * kHiddenSize * 2];
        u_ptr[j * 4 + 3] = u_ptr_fuse[j + kHiddenSize * kHiddenSize * 3];
      }
      for (int j = 0; j < kHiddenSize; ++j) {
        b_ptr[j * 4] = b_ptr_fuse[j];
        b_ptr[j * 4 + 1] = b_ptr_fuse[j + kHiddenSize];
        b_ptr[j * 4 + 2] = b_ptr_fuse[j + kHiddenSize * 2];
        b_ptr[j * 4 + 3] = b_ptr_fuse[j + kHiddenSize * 3];
      }
      LSTMHostCellParams_75_7 param = {
          lstmInputDev, all_zero_state, all_zero_state, w_ptr, u_ptr, b_ptr};
      lstmInitParams.push_back(param);
    }

    input_ = reinterpret_cast<Tensor<kBatch * kInputStep * 171> *>(
        reinterpret_cast<float *>(gpu_memory_) + state_size);
    lstms.init(lstmInitParams);
  }
  virtual void TearDown() override {
    if (gpu_memory_) {
      cudaFree(gpu_memory_);
    }
  }
  std::unique_ptr<DeepSpeech2Model<kBatch, kOutputStep, kNumClass>> model_;
  std::vector<float> pytorch_output_;
  std::vector<float> pytorch_input_;
  Tensor<kBatch * kInputStep * 171> *input_;
  void *gpu_memory_;
};

TEST_F(DeepSpeech2Test, TestSingleInfer) {
  enum { kLoop = 1000, kWarmupLoop = 100 };

  std::vector<float> output_test(kBatch * kOutputStep * kNumClass);
  std::vector<float> output(kBatch * kOutputStep * kNumClass);
  for (unsigned i = 0; i < kWarmupLoop; ++i) {
    CUDA_CHECK(cudaMemcpy(&input_->data, pytorch_input_.data(),
                          pytorch_input_.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    model_->Forward(input_);
    if (i == 0) {
      auto device_output = model_->GetOutput();
      static_assert(sizeof(*device_output) ==
                    kBatch * kOutputStep * kNumClass * sizeof(float));
      CUDA_CHECK(cudaMemcpy(output_test.data(), device_output,
                            sizeof(*device_output), cudaMemcpyDeviceToHost));
    }
  };
  CUDA_CHECK(cudaMemcpy(&input_->data, pytorch_input_.data(),
                        pytorch_input_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  timeval time_start;
  timeval time_end;
  float walltimes = 0.00000f;

  for (int i = 0; i < kLoop; i++) {
    // Different, input/output memcpy time
    gettimeofday(&time_start, NULL);
    model_->Forward(input_);
    gettimeofday(&time_end, NULL);
    float once_time = (time_end.tv_sec - time_start.tv_sec) * 1000000 +
                      time_end.tv_usec - time_start.tv_usec;
    once_time /= 1000;
    walltimes += once_time;
  }
  printf("Average time (ms): %f\n", walltimes / kLoop);
  auto device_output = model_->GetOutput();
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, sizeof(*device_output),
                        cudaMemcpyDeviceToHost));

  ASSERT_EQ(output_test.size(), pytorch_output_.size());
  for (unsigned i = 0; i < output_test.size(); ++i) {
    float diff = fabs(output_test[i] - pytorch_output_[i]);
    ASSERT_LT(diff, 1e-3);
  }
}
} // namespace mica::experiments::lstm
