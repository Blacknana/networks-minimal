#pragma once

#include "cuda_ops.h"
#include "op.h"

template <uint64_t kBatchSize, uint64_t kInSize, uint64_t kOutSize,
          bool kIsBias, bool kFuseRelu = false, bool kFuseBatchNorm1d = false>
class Linear : public Op<kBatchSize * kOutSize> {
public:
  explicit Linear(void *device_ptr) {
    state_ = reinterpret_cast<decltype(state_)>(device_ptr);
    Parent::output_ = &state_->output;
  }

  static uint64_t GetStateSize() { return sizeof(*state_) / sizeof(float); }

  template <int T> void Forward(Tensor<T> *input) {
    static_assert(T == kBatchSize * kInSize);

    dim3 dim_block(kTileSize, kTileSize);

    dim3 dim_grid((kOutSize + kTileSize - 1) / kTileSize,
                  (kBatchSize + kTileSize - 1) / kTileSize, 1);

    void *weight_ptr = &(state_->weight);
    void *output_ptr = &(state_->output);

    if constexpr (kIsBias) {
      void *bias_ptr = &(state_->bias);
      if constexpr (kFuseBatchNorm1d) {
        void *bn_ptr = &(state_->bn_param);
        void *args[] = {&input, &weight_ptr, &bias_ptr, &bn_ptr, &output_ptr};
        cudaLaunchKernel(
            (const void *)
                operator_fuse_bn1d_linear_bias_h<kBatchSize, kInSize, kOutSize>,
            dim_grid, dim_block, (void **)args, kTileSize * kTileSize * 2);
      }
    }
    CUDA_POST_KERNEL_CHECK;
  }

  virtual std::vector<std::string_view> GetParameters() override {
    return {std::string_view(reinterpret_cast<char *>(&state_->output) +
                                 sizeof(state_->output),
                             sizeof(*state_) - sizeof(state_->output))};
  }

private:
  using Parent = Op<kBatchSize * kOutSize>;
  LinearState<kBatchSize, kInSize, kOutSize, kIsBias, kFuseBatchNorm1d> *state_;
};