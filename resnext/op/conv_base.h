#pragma once

#include "cuda_ops.h"
#include "op.h"

template <uint64_t kBatchSize, uint64_t kChannels, uint64_t kHeight,
          uint64_t kWidth, uint64_t kKernelH, uint64_t kKernelW, uint64_t kPadH,
          uint64_t kPadW, uint64_t kStrideH, uint64_t kStrideW>
static constexpr uint64_t ConvArgSize() {
  constexpr uint64_t pooled_height =
      (kHeight + 2 * kPadH - kKernelH) / kStrideH + 1;
  constexpr uint64_t pooled_width =
      (kWidth + 2 * kPadW - kKernelW) / kStrideW + 1;
  return kBatchSize * kChannels * pooled_height * pooled_width;
}

template <uint64_t kBatchSize, uint64_t kHeight, uint64_t kWidth,
          uint64_t kChannelIn, uint64_t kChannelOut, uint64_t kKernelH,
          uint64_t kKernelW, uint64_t kPadH, uint64_t kPadW, uint64_t kStrideH,
          uint64_t kStrideW, bool kIsBias, bool kFuseBN = false,
          bool kFuseRelu = kFuseBN, uint64_t kGroupSize = 1,
          bool kFuseRelu6 = false>
class Conv
    : public Op<ConvArgSize<kBatchSize, kChannelOut, kHeight, kWidth, kKernelH,
                            kKernelW, kPadH, kPadW, kStrideH, kStrideW>()> {
public:
  explicit Conv(void *device_ptr) {
    state_ = reinterpret_cast<decltype(state_)>(device_ptr);
    Parent::output_ = &state_->output;
  }

  static uint64_t GetStateSize() { return sizeof(*state_) / sizeof(float); };

  template <int T> void Forward(Tensor<T> *input) {
    static_assert(T == kBatchSize * kChannelIn * kHeight * kWidth);
    static_assert(kChannelIn % kGroupSize == 0 &&
                  kChannelOut % kGroupSize == 0);

    // im2col
    // [kBatchSize*(C_in*k_h*k_w)*(height_col * width_col)]
    constexpr uint64_t size =
        ConvArgSize<1, kChannelIn / kGroupSize, kHeight, kWidth, kKernelH,
                    kKernelW, kPadH, kPadW, kStrideH, kStrideW>();

    dim3 im2col_dim_grid((size + kBlockSize - 1) / kBlockSize,
                         kBatchSize * kGroupSize);

    void *col_ptr = &state_->col;
    void *im2col_args[] = {(void *)&input, &col_ptr};
    cudaLaunchKernel(
        (const void *)
            im2col_h<kBatchSize * kGroupSize, kHeight, kWidth,
                     kChannelIn / kGroupSize, kChannelOut, kKernelH, kKernelW,
                     kPadH, kPadW, kStrideH, kStrideW, kIsBias, T, kColSize>,
        im2col_dim_grid, dim3(kBlockSize), (void **)im2col_args, 0);

    CUDA_POST_KERNEL_CHECK;

    // Y = F * col
    // [C_out*(C_in*k_h*k_w)] * [kBatchSize *
    // (C_in*k_h*k_w)*(height_col*width_col)] = [kBatchSize * kChannelOut *
    // (height_col * width_col)]

    dim3 dim_block(kTileSize, kTileSize);

    constexpr uint64_t kHeightOut =
        (kHeight + 2 * kPadH - kKernelH) / kStrideH + 1;
    constexpr uint64_t kWidthOut =
        (kWidth + 2 * kPadW - kKernelW) / kStrideW + 1;

    constexpr uint64_t kMatmulHeight = kChannelOut / kGroupSize;
    constexpr uint64_t kMatmulK = kChannelIn / kGroupSize * kKernelH * kKernelW;
    constexpr uint64_t kMatmulWidth = kWidthOut * kHeightOut;

    dim3 matmul_dim_grid((kMatmulWidth + kTileSize - 1) / kTileSize,
                         (kMatmulHeight + kTileSize - 1) / kTileSize,
                         kBatchSize * kGroupSize);

    void *filter_ptr = &state_->filter;
    void *output_ptr = &state_->output;

    if constexpr (kFuseBN) {
      static_assert(kIsBias == false);
      void *bn_ptr = &state_->bn_param;
      void *fuse_args[] = {&filter_ptr, &col_ptr, &output_ptr, &bn_ptr};
      cudaLaunchKernel(
          (const void *)operator_conv_h<
              kMatmulHeight, kMatmulK, kMatmulWidth, 1, 1, kBatchSize,
              kChannelOut, kHeightOut, kWidthOut, kFuseRelu, kGroupSize>,
          matmul_dim_grid, dim_block, (void **)fuse_args,
          kTileSize * kTileSize * 2);
      cudaLaunchKernel(
          (const void *)operator_fuse_bn_relu_h<
              kMatmulHeight, kMatmulK, kMatmulWidth, 1, 1, kBatchSize,
              kChannelOut, kHeightOut, kWidthOut, kFuseRelu, kGroupSize>,
          matmul_dim_grid, dim_block, (void **)fuse_args,
          kTileSize * kTileSize * 2);
    }

    CUDA_POST_KERNEL_CHECK;
  }

  virtual std::vector<std::string_view> GetParameters() override {
    if constexpr (kIsBias) {
      if constexpr (kFuseBN) {
        return {std::string_view(reinterpret_cast<char *>(&state_->filter),
                                 sizeof(state_->filter) + sizeof(state_->bias) +
                                     sizeof(state_->bn_param))};
      } else {
        return {
            std::string_view(reinterpret_cast<char *>(&state_->filter),
                             sizeof(state_->filter) + sizeof(state_->bias))};
      }
    } else {
      if constexpr (kFuseBN) {
        return {std::string_view(reinterpret_cast<char *>(&state_->filter),
                                 sizeof(state_->filter) +
                                     sizeof(state_->bn_param))};
      } else {
        return {std::string_view(reinterpret_cast<char *>(&state_->filter),
                                 sizeof(state_->filter))};
      }
    }
    return {};
  }

private:
  static constexpr uint64_t kOutputSize =
      ConvArgSize<kBatchSize, kChannelOut, kHeight, kWidth, kKernelH, kKernelW,
                  kPadH, kPadW, kStrideH, kStrideW>();
  static constexpr uint64_t kFilterSize =
      kChannelOut * kChannelIn / kGroupSize * kKernelH * kKernelW;
  static constexpr uint64_t kBiasSize = kChannelOut * kIsBias;
  static constexpr uint64_t kColSize =
      ConvArgSize<kBatchSize, kChannelIn, kHeight, kWidth, kKernelH, kKernelW,
                  kPadH, kPadW, kStrideH, kStrideW>() *
      kKernelH * kKernelW;

  using Parent = Op<kOutputSize>;
  ConvState<kOutputSize, kChannelOut, kFilterSize, kColSize, kBiasSize, kIsBias,
            kFuseBN> *state_;
};