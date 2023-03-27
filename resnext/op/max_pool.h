#pragma once

#include "cuda_ops.h"
#include "op.h"

template <uint64_t kBatchSize, uint64_t kChannels, uint64_t kHeight,
          uint64_t kWidth, uint64_t kKernelH, uint64_t kKernelW, uint64_t kPadH,
          uint64_t kPadW, uint64_t kStrideH, uint64_t kStrideW>
static constexpr uint64_t MaxPoolArgSize() {
  constexpr uint64_t pooled_height =
      (kHeight + 2 * kPadH - kKernelH) / kStrideH + 1;
  constexpr uint64_t pooled_width =
      (kWidth + 2 * kPadW - kKernelW) / kStrideW + 1;
  return kBatchSize * kChannels * pooled_height * pooled_width;
}

template <uint64_t kBatchSize, uint64_t kChannels, uint64_t kHeight,
          uint64_t kWidth, uint64_t kKernelH, uint64_t kKernelW, uint64_t kPadH,
          uint64_t kPadW, uint64_t kStrideH, uint64_t kStrideW>
class MaxPool
    : public Op<MaxPoolArgSize<kBatchSize, kChannels, kHeight, kWidth, kKernelH,
                               kKernelW, kPadH, kPadW, kStrideH, kStrideW>()> {
public:
  explicit MaxPool(void *device_ptr) {
    state_ = reinterpret_cast<decltype(state_)>(device_ptr);
    Parent::output_ = &state_->output;
  };

  static uint64_t GetStateSize() { return sizeof(*state_) / sizeof(float); };

  template <int T> void Forward(Tensor<T> *input) {
    static_assert(T == kBatchSize * kChannels * kHeight * kWidth);

    constexpr uint64_t grid_size = (kSize + kBlockSize - 1) / kBlockSize;

    void *args[] = {&input, &state_};

    cudaLaunchKernel(
        (const void *)operator_max_pool_h<kBatchSize, kChannels, kHeight,
                                          kWidth, kKernelH, kKernelW, kPadH,
                                          kPadW, kStrideH, kStrideW, T, kSize>,
        dim3(grid_size), dim3(kBlockSize), (void **)args, 0);

    CUDA_POST_KERNEL_CHECK;
  };

  virtual std::vector<std::string_view> GetParameters() override { return {}; }

private:
  static constexpr uint64_t kSize =
      MaxPoolArgSize<kBatchSize, kChannels, kHeight, kWidth, kKernelH, kKernelW,
                     kPadH, kPadW, kStrideH, kStrideW>();
  using Parent = Op<kSize>;
  MaxPoolState<kSize> *state_;
};
