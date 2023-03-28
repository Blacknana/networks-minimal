#include "op/avg_pool.h"
#include "op/bottleneck_add.h"
#include "op/conv_base.h"
#include "op/flatten.h"
#include "op/input.h"
#include "op/linear.h"
#include "op/max_pool.h"
#include "op/op_helper.h"
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <string_view>

template <uint32_t kBatchSize, uint32_t kHeight, uint32_t kWidth,
          uint32_t kChannelIn, uint32_t kChannelBase, bool kNeedDownsample,
          uint32_t kStride = 1, uint32_t kGroupSize = 1,
          uint32_t kWidthPerGroup = 64>
class BasicBlock : public Op<kBatchSize * kChannelBase * 4 *
                             (kHeight / kStride) * (kWidth / kStride)> {
public:
  BasicBlock(void *device_ptr) {
    float *mem = reinterpret_cast<float *>(device_ptr);
    auto set_mem = [&mem](auto &op) {
      using T = typename std::remove_pointer<decltype(op.get())>::type;
      op.reset(new T(mem));
      mem += T::GetStateSize();
    };

    if constexpr (kNeedDownsample) {
      op_map(set_mem, fuse_conv1_bn_relu_, fuse_conv2_bn_relu_, fuse_conv3_bn_,
             downsample_conv_bn_, fuse_add_relu_);
    } else {
      op_map(set_mem, fuse_conv1_bn_relu_, fuse_conv2_bn_relu_, fuse_conv3_bn_,
             fuse_add_relu_);
    }

    Parent::output_ = fuse_add_relu_->GetOutput();
  }

  static uint64_t GetStateSize() {
    uint64_t state_size =
        decltype(fuse_conv1_bn_relu_)::element_type::GetStateSize() +
        decltype(fuse_conv2_bn_relu_)::element_type::GetStateSize() +
        decltype(fuse_conv3_bn_)::element_type::GetStateSize() +
        decltype(fuse_add_relu_)::element_type::GetStateSize();

    if constexpr (kNeedDownsample) {
      state_size += decltype(downsample_conv_bn_)::element_type::GetStateSize();
    }

    return state_size;
  }

  template <int T> void Forward(Tensor<T> *input) {
    static_assert(T == kBatchSize * kChannelIn * kHeight * kWidth);
    auto seq_model = [](auto &arg, auto &op) {
      op->Forward(arg);
      return op->GetOutput();
    };

    Tensor<kSize> *add_input0 =
        op_reduce(seq_model, input, fuse_conv1_bn_relu_.get(),
                  fuse_conv2_bn_relu_.get(), fuse_conv3_bn_.get());
    Tensor<kSize> *add_input1;
    if constexpr (kNeedDownsample) {
      add_input1 = op_reduce(seq_model, input, downsample_conv_bn_.get());
    } else {
      add_input1 = input;
    }
    fuse_add_relu_->Forward(add_input0, add_input1);
  }

  virtual std::vector<std::string_view> GetParameters() override {
    std::vector<std::string_view> params;
    auto get_param = [&params](auto &op) {
      auto op_param = op->GetParameters();
      if (op_param.size()) {
        params.insert(params.end(), op_param.begin(), op_param.end());
      }
    };

    if constexpr (kNeedDownsample) {
      op_map(get_param, fuse_conv1_bn_relu_, fuse_conv2_bn_relu_,
             fuse_conv3_bn_, downsample_conv_bn_, fuse_add_relu_);
    } else {
      op_map(get_param, fuse_conv1_bn_relu_, fuse_conv2_bn_relu_,
             fuse_conv3_bn_, fuse_add_relu_);
    }
    return params;
  }

private:
  constexpr static uint32_t kChannelOut = 4 * kChannelBase;
  constexpr static uint32_t kChannelConv1 =
      kChannelBase * kWidthPerGroup / 64 * kGroupSize;
  constexpr static uint32_t kHeightOut = kHeight / kStride;
  constexpr static uint32_t kWidthOut = kWidth / kStride;
  constexpr static uint64_t kSize =
      kBatchSize * kChannelOut * kHeightOut * kWidthOut;
  using Parent = Op<kSize>;

  std::unique_ptr<Conv<kBatchSize, kHeight, kWidth, kChannelIn, kChannelConv1,
                       1, 1, 0, 0, 1, 1, false, true>>
      fuse_conv1_bn_relu_;
  std::unique_ptr<
      Conv<kBatchSize, kHeight, kWidth, kChannelConv1, kChannelConv1, 3, 3, 1,
           1, kStride, kStride, false, true, true, kGroupSize>>
      fuse_conv2_bn_relu_;
  std::unique_ptr<Conv<kBatchSize, kHeightOut, kWidthOut, kChannelConv1,
                       kChannelOut, 1, 1, 0, 0, 1, 1, false, true, false>>
      fuse_conv3_bn_;
  std::unique_ptr<Conv<kBatchSize, kHeight, kWidth, kChannelIn, kChannelOut, 1,
                       1, 0, 0, kStride, kStride, false, true, false>>
      downsample_conv_bn_;
  std::unique_ptr<
      BottleneckAdd<kBatchSize, kChannelOut, kHeightOut, kWidthOut, true>>
      fuse_add_relu_;
};

template <uint32_t kBatchSize, uint32_t kHeight, uint32_t kWidth,
          uint32_t kChannelIn, uint32_t kChannelBase, uint32_t kNumber,
          uint32_t kStride = 1, uint32_t kGroupSize = 1,
          uint32_t kWidthPerGroup = 64>
class BottleNeck : public Op<kBatchSize * kChannelBase * 4 *
                             (kHeight / kStride) * (kWidth / kStride)> {
public:
  BottleNeck(void *device_ptr) {
    static_assert(kNumber >= 2);

    float *mem = reinterpret_cast<float *>(device_ptr);

    block1_.reset(
        new (typename std::remove_pointer<decltype(block1_.get())>::type)(mem));
    mem += std::remove_pointer<decltype(block1_.get())>::type::GetStateSize();

    for (unsigned i = 0; i < kNumber - 1; ++i) {
      using T =
          typename std::remove_pointer<decltype(blocks_.at(i).get())>::type;
      blocks_.at(i).reset(new T(mem));
      mem += T::GetStateSize();
    }

    Parent::output_ = blocks_.back()->GetOutput();
  }

  static uint64_t GetStateSize() {
    uint64_t state_size = 0;
    state_size +=
        std::remove_pointer<decltype(block1_.get())>::type::GetStateSize();

    for (unsigned i = 0; i < kNumber - 1; ++i) {
      using T =
          typename std::remove_pointer<decltype(blocks_.at(i).get())>::type;
      state_size += T::GetStateSize();
    }
    return state_size;
  }

  template <int T> void Forward(Tensor<T> *input) {
    block1_->Forward(input);
    for (unsigned i = 0; i < kNumber - 1; ++i) {
      if (i == 0) {
        auto output = block1_->GetOutput();
        blocks_.at(i)->Forward(output);
      } else {
        auto output = blocks_.at(i - 1)->GetOutput();
        blocks_.at(i)->Forward(output);
      }
    }
  }

  virtual std::vector<std::string_view> GetParameters() override {
    auto params = block1_->GetParameters();

    for (unsigned i = 0; i < kNumber - 1; ++i) {
      auto op_param = blocks_.at(i)->GetParameters();
      params.insert(params.end(), op_param.begin(), op_param.end());
    }

    return params;
  }

private:
  constexpr static uint32_t kChannelOut = 4 * kChannelBase;
  constexpr static uint32_t kHeightOut = kHeight / kStride;
  constexpr static uint32_t kWidthOut = kWidth / kStride;
  constexpr static uint64_t kSize =
      kBatchSize * kChannelOut * kHeightOut * kWidthOut;
  using Parent = Op<kSize>;

  std::unique_ptr<
      BasicBlock<kBatchSize, kHeight, kWidth, kChannelIn, kChannelBase, true,
                 kStride, kGroupSize, kWidthPerGroup>>
      block1_;
  std::array<std::unique_ptr<BasicBlock<kBatchSize, kHeightOut, kWidthOut,
                                        kChannelOut, kChannelBase, false, 1,
                                        kGroupSize, kWidthPerGroup>>,
             kNumber - 1>
      blocks_;
};

template <uint32_t kBatchSize, uint32_t kNumClass, uint32_t kGroupSize = 1,
          uint32_t kWidthPerGroup = 64>
class Resnet50Model : public Op<kBatchSize * kNumClass> {
public:
  Resnet50Model(void *device_ptr) {
    float *mem = reinterpret_cast<float *>(device_ptr);
    auto set_mem = [&mem](auto &op) {
      using T = typename std::remove_pointer<decltype(op.get())>::type;
      op.reset(new T(mem));
      mem += T::GetStateSize();
    };

    op_map(set_mem, fuse_conv1_bn_relu_, maxpool1_, conv2_bottleneck_,
           conv3_bottleneck_, conv4_bottleneck_, conv5_bottleneck_, avgpool1_,
           flatten1_, fc1_);
    Parent::output_ = fc1_->GetOutput();
  }

  template <int T> void Forward(Tensor<T> *input) {
    static_assert(T == kInputSize);
    auto seq_model = [](auto &arg, auto &op) {
      op->Forward(arg);
      return op->GetOutput();
    };

    op_reduce(seq_model, input, fuse_conv1_bn_relu_.get(), maxpool1_.get(),
              conv2_bottleneck_.get(), conv3_bottleneck_.get(),
              conv4_bottleneck_.get(), conv5_bottleneck_.get(), avgpool1_.get(),
              flatten1_.get(), fc1_.get());
  }

  static uint64_t GetStateSize() {
    uint64_t state_size =
        decltype(fuse_conv1_bn_relu_)::element_type::GetStateSize() +
        decltype(maxpool1_)::element_type::GetStateSize() +
        decltype(conv2_bottleneck_)::element_type::GetStateSize() +
        decltype(conv3_bottleneck_)::element_type::GetStateSize() +
        decltype(conv4_bottleneck_)::element_type::GetStateSize() +
        decltype(conv5_bottleneck_)::element_type::GetStateSize() +
        decltype(avgpool1_)::element_type::GetStateSize() +
        decltype(flatten1_)::element_type::GetStateSize() +
        decltype(fc1_)::element_type::GetStateSize();

    return state_size;
  }

  virtual std::vector<std::string_view> GetParameters() override {
    std::vector<std::string_view> params;
    auto get_param = [&params](auto &op) {
      auto op_param = op->GetParameters();
      if (op_param.size()) {
        params.insert(params.end(), op_param.begin(), op_param.end());
      }
    };

    op_map(get_param, fuse_conv1_bn_relu_, maxpool1_, conv2_bottleneck_,
           conv3_bottleneck_, conv4_bottleneck_, conv5_bottleneck_, avgpool1_,
           flatten1_, fc1_);

    return params;
  }

private:
  std::unique_ptr<
      Conv<kBatchSize, 224, 224, 3, 64, 7, 7, 3, 3, 2, 2, false, true>>
      fuse_conv1_bn_relu_;
  std::unique_ptr<MaxPool<kBatchSize, 64, 112, 112, 3, 3, 1, 1, 2, 2>>
      maxpool1_;
  std::unique_ptr<
      BottleNeck<kBatchSize, 56, 56, 64, 64, 3, 1, kGroupSize, kWidthPerGroup>>
      conv2_bottleneck_;
  std::unique_ptr<BottleNeck<kBatchSize, 56, 56, 256, 128, 4, 2, kGroupSize,
                             kWidthPerGroup>>
      conv3_bottleneck_;
  std::unique_ptr<BottleNeck<kBatchSize, 28, 28, 512, 256, 6, 2, kGroupSize,
                             kWidthPerGroup>>
      conv4_bottleneck_;
  std::unique_ptr<BottleNeck<kBatchSize, 14, 14, 1024, 512, 3, 2, kGroupSize,
                             kWidthPerGroup>>
      conv5_bottleneck_;
  std::unique_ptr<AvgPool<kBatchSize, 2048, 7, 7, 7, 7, 0, 0, 7, 7>> avgpool1_;
  std::unique_ptr<Flatten<kBatchSize, 2048, 1, 1>> flatten1_;
  std::unique_ptr<Linear<kBatchSize, 2048, kNumClass, true>> fc1_;
  constexpr static uint64_t kInputSize = kBatchSize * 224 * 224 * 3;
  constexpr static uint64_t kSize = kBatchSize * kNumClass;
  using Parent = Op<kSize>;
};
