#pragma once

#include "cuda_ops.h"
#include "op.h"
#include "rammer_ops.h"
#include "utils.h"

template <int kBatchSize, int kChannels, int kHeight, int kWidth, int kPaddingH,
          int kPaddingW>
class Padding;

template <> class Padding<1, 32, 86, 150, 5, 10> : public Op<32 * 96 * 170> {
public:
  Padding(void *device_ptr) {
    state_ = reinterpret_cast<PaddingState *>(device_ptr);
    CUDA_CHECK(cudaMemset(state_, 0, sizeof(*state_)));
    Parent::output_ = &state_->output;
  };

  void Forward(Tensor<32 * 86 * 150> *input) {
    void *args[] = {&input, &state_};
    cudaLaunchKernel((const void *)RammerPadding, dim3(86, 15, 1),
                     dim3(32, 10, 1), (void **)args, 0);
    CUDA_POST_KERNEL_CHECK;
  };

  static uint64_t GetStateSize() { return kSize; };

  virtual std::vector<std::string_view> GetParameters() override { return {}; }

private:
  static constexpr uint64_t kSize = 32 * 96 * 170;
  PaddingState *state_;

  using Parent = Op<kSize>;
};