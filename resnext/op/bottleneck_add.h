#pragma once

#include "cuda_ops.h"
#include "op.h"

template <int kBatchSize, int kChannels, int kHeight, int kWidth,
          bool kFuseRelu>
class BottleneckAdd : public Op<kBatchSize * kChannels * kHeight * kWidth> {
public:
  BottleneckAdd(void *device_ptr) {
    Parent::output_ = reinterpret_cast<Tensor<kSize> *>(device_ptr);
  }

  template <int T> void Forward(Tensor<T> *input1, Tensor<T> *input2) {
    static_assert(T == kSize);
    uint64_t grid_size = (T + kBlockSize - 1) / kBlockSize;
    void *output = Parent::output_;
    void *args[] = {&input1, &input2, &output};
    cudaLaunchKernel((const void *)operator_vecaddvec_h<T, kFuseRelu>,
                     dim3(grid_size), dim3(kBlockSize), (void **)args, 0);
  }

  virtual std::vector<std::string_view> GetParameters() override { return {}; }

  static uint64_t GetStateSize() { return kSize; };

private:
  static constexpr uint64_t kSize = kBatchSize * kChannels * kHeight * kWidth;
  using Parent = Op<kSize>;
};
