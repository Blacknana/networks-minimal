#pragma once

#include "op.h"
#include "utils.h"
#include <memory>

template <int kBatchSize, int kChannels, int kHeight, int kWidth>
class Input : public Op<kBatchSize * kChannels * kHeight * kWidth> {
public:
  Input(void *device_ptr) {
    Parent::output_ = reinterpret_cast<Tensor<kSize> *>(device_ptr);
  };
  static uint64_t GetStateSize() { return kSize; };
  void Forward(const std::vector<float> &data) {
    auto dst = Parent::GetOutput();
    ASSERT(data.size() == kSize);
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(dst), data.data(),
                          data.size() * sizeof(float), cudaMemcpyHostToDevice));
  };

  virtual std::vector<std::string_view> GetParameters() override { return {}; }

private:
  using Parent = Op<kBatchSize * kChannels * kHeight * kWidth>;
  constexpr static uint64_t kSize = kBatchSize * kChannels * kHeight * kWidth;
};