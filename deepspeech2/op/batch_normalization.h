#pragma once

#include <algorithm>
#include <cfloat>
#include <iostream>

#include "cuda_ops.h"
#include "op.h"
#include <cuda_runtime.h>

template <uint64_t kBatchSize, uint64_t kChannel, uint64_t kHeight,
          uint64_t kWidth>
class BatchNormalization : public Op<kBatchSize * kChannel * kHeight * kWidth> {
public:
  explicit BatchNormalization(void *device_ptr) {
    state_ = reinterpret_cast<decltype(state_)>(device_ptr);
    Parent::output_ = &state_->output;
  };

  template <int T> void Forward(Tensor<T> *input) {
    static_assert(T == kSize);
    constexpr uint64_t kGridSize = (T + kBlockSize - 1) / kBlockSize;
    void *args[] = {&input, &state_};

    cudaLaunchKernel(
        (const void *)operator_batch_normalization_h<kBatchSize, kChannel,
                                                     kHeight, kWidth, kSize>,
        dim3(kGridSize), dim3(kBlockSize), (void **)args, 0);
    CUDA_POST_KERNEL_CHECK;
  }

  static uint64_t GetStateSize() { return sizeof(*state_) / sizeof(float); };

  virtual std::vector<std::string_view> GetParameters() override {
    return {std::string_view(reinterpret_cast<char *>(&state_->output) +
                                 sizeof(state_->output),
                             sizeof(*state_) - sizeof(state_->output))};
  }

private:
  static constexpr uint64_t kSize = kBatchSize * kChannel * kHeight * kWidth;
  using Parent = Op<kSize>;
  BatchNormState<kSize, kChannel> *state_;
};