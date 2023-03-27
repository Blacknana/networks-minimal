#pragma once

#include "cuda_ops.h"
#include "op.h"
#include "utils.h"

template <int kBatchSize, int kChannels, int kHeight, int kWidth,
          int kInplace = 0, bool kReLU6 = false>
class ReLU : public Op<kBatchSize * kChannels * kHeight * kWidth> {
public:
  ReLU(void *device_ptr) {
    if constexpr (!kInplace) {
      Parent::output_ = reinterpret_cast<Tensor<kSize> *>(device_ptr);
    }
  };

  template <int T> void Forward(Tensor<T> *input) {
    static_assert(T == kSize);
    if constexpr (kInplace) {
      Parent::output_ = input;
    }

    uint64_t grid_size = (T + kBlockSize - 1) / kBlockSize;
    void *output = Parent::output_;
    void *args[] = {&input, &output};
    cudaLaunchKernel((const void *)operator_vectorrelu_h<kSize, kReLU6>,
                     dim3(grid_size), dim3(kBlockSize), (void **)args, 0);
    CUDA_POST_KERNEL_CHECK;
  };

  static uint64_t GetStateSize() { return kInplace ? 0 : kSize; };

  virtual std::vector<std::string_view> GetParameters() override { return {}; }

private:
  static constexpr uint64_t kSize = kBatchSize * kChannels * kHeight * kWidth;
  using Parent = Op<kSize>;
};