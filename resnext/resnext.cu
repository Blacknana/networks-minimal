#include "op/utils.h"
#include "resnext.h"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string_view>

#include <fstream>
#include <random>
#include <string>
#include <vector>

enum {
  kBatch = 1,
  kNumClass = 1000,
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

class Resnext50_32x4dTest : public ::testing::Test {
protected:
  virtual void SetUp() override {

    const std::string &kDataPath = ".";
    const std::string &kOutputFile = "resnext50_32x4d.output";

    gpu_memory_ = nullptr;
    pytorch_input_ = RandomData(kBatch * 3 * 224 * 224);
    pytorch_output_ = ReadAll(kOutputFile, kDataPath);
    auto state_size = Resnet50Model<kBatch, kNumClass, 32, 4>::GetStateSize();
    CUDA_CHECK(cudaMalloc(&gpu_memory_, (state_size + pytorch_input_.size()) *
                                            sizeof(float)));
    model_.reset(new Resnet50Model<kBatch, kNumClass, 32, 4>(gpu_memory_));

    auto state = RandomData(state_size);
    CUDA_CHECK(cudaMemcpy(gpu_memory_, state.data(), state_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    input_ = reinterpret_cast<Tensor<kBatch * 224 * 224 * 3> *>(
        reinterpret_cast<float *>(gpu_memory_) + state_size);
  }
  virtual void TearDown() override {
    if (gpu_memory_) {
      cudaFree(gpu_memory_);
    }
  }
  std::unique_ptr<Resnet50Model<kBatch, kNumClass, 32, 4>> model_;
  std::vector<float> pytorch_output_;
  std::vector<float> pytorch_input_;
  Tensor<kBatch * 224 * 224 * 3> *input_;
  void *gpu_memory_;
};

TEST_F(Resnext50_32x4dTest, TestSingleInfer) {
  enum {
    kLoop = 100,
  };

  std::vector<float> output(kBatch * kNumClass);

  for (unsigned i = 0; i < kLoop; ++i) {
    CUDA_CHECK(cudaMemcpy(&input_->data, pytorch_input_.data(),
                          pytorch_input_.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    model_->Forward(input_);
    auto device_output = model_->GetOutput();
    static_assert(sizeof(*device_output) == kNumClass * sizeof(float) * kBatch);
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, sizeof(*device_output),
                          cudaMemcpyDeviceToHost));
  };

  cudaEvent_t start_cuda, stop_cuda;
  cudaEventCreate(&start_cuda);
  cudaEventCreate(&stop_cuda);
  cudaEventRecord(start_cuda, 0);

  for (unsigned i = 0; i < kLoop; ++i) {
    CUDA_CHECK(cudaMemcpy(&input_->data, pytorch_input_.data(),
                          pytorch_input_.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    model_->Forward(input_);
    auto device_output = model_->GetOutput();
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, sizeof(*device_output),
                          cudaMemcpyDeviceToHost));
  }

  cudaEventRecord(stop_cuda, 0);
  cudaEventSynchronize(stop_cuda);
  float cuda_event_timepassed = 0;
  cudaEventElapsedTime(&cuda_event_timepassed, start_cuda, stop_cuda);
  printf("Elapsed time (ms): %f\n", cuda_event_timepassed / kLoop);

  ASSERT_EQ(output.size(), pytorch_output_.size());
  for (unsigned i = 0; i < output.size(); ++i) {
    float diff = fabs(output[i] - pytorch_output_[i]);
    ASSERT_LT(diff, 0.0001);
  }

  cudaEventDestroy(start_cuda);
  cudaEventDestroy(stop_cuda);
}
