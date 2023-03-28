#include "lstm/lstm.h"
#include "op/batch_normalization.h"
#include "op/conv.h"
#include "op/input.h"
#include "op/linear.h"
#include "op/op.h"
#include "op/op_helper.h"
#include "op/padding.h"
#include "op/permute.h"
#include "op/relu.h"
#include "op/utils.h"
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <string_view>
#include <sys/time.h>

namespace mica::experiments::lstm {
template <uint32_t kBatchSize, uint32_t kOutputStep, uint32_t kNumClass>
class DeepSpeech2Model : public Op<kOutputStep * kBatchSize * kNumClass> {
public:
  enum {
    kInputHeight = 171,
    kInputWidth = 300,

    kConv1FilterH = 11,
    kConv1FilterW = 41,
    kConv1PaddingH = 5,
    kConv1PaddingW = 20,
    kConv1StrideHW = 2,
    kConv1ChannelOut = 32,
    kConv1OutputH =
        (kInputHeight + 2 * kConv1PaddingH - kConv1FilterH) / kConv1StrideHW +
        1,
    kConv1OutputW =
        (kInputWidth + 2 * kConv1PaddingW - kConv1FilterW) / kConv1StrideHW + 1,

    kConv2FilterH = 11,
    kConv2FilterW = 21,
    kConv2PaddingH = 5,
    kConv2PaddingW = 10,
    kConv2StrideH = 1,
    kConv2StrideW = 2,
    kConv2ChannelOut = 32,
    kConv2OutputH =
        (kConv1OutputH + 2 * kConv2PaddingH - kConv2FilterH) / kConv2StrideH +
        1,
    kConv2OutputW =
        (kConv1OutputW + 2 * kConv2PaddingW - kConv2FilterW) / kConv2StrideW +
        1,

    kLstmInputSize = kConv2ChannelOut * kConv2OutputH,
    kLstmInputStep = kConv2OutputW,
    kHiddenSize = 256,
    kLstmLayer = 7,
  };

  DeepSpeech2Model(void *device_ptr) {
    float *mem = reinterpret_cast<float *>(device_ptr);

    auto set_mem = [&mem](auto &op) {
      using T = typename std::remove_pointer<decltype(op.get())>::type;
      op.reset(new T(mem));
      mem += T::GetStateSize();
    };

    op_map(set_mem, fuse_conv1_relu6_bn_, pad_, conv2_, relu6_, bn_, permute_,
           fuse_bn1d_fc_); //, lstm1_, lstm2_,

    lstm_output = mem + kInputHeight * kInputWidth * kBatchSize;
    Parent::output_ = fuse_bn1d_fc_->GetOutput();
  }

  template <int T> void Forward(Tensor<T> *input) {
    static_assert(T == kInputSize);

    auto seq_model = [](auto &arg, auto &op) {
      op->Forward(arg);
      return op->GetOutput();
    };

    op_reduce(seq_model, input, fuse_conv1_relu6_bn_.get(), pad_.get(),
              conv2_.get(), relu6_.get(), bn_.get(), permute_.get());

    float *permute_output = reinterpret_cast<float *>(permute_->GetOutput());

    lstms_.compute();
    lstm_output = lstms_.getLastCellAllStep();

    fuse_bn1d_fc_->Forward(
        reinterpret_cast<Tensor<kBatchSize * kLstmInputStep * kHiddenSize> *>(
            lstm_output));
    cudaDeviceSynchronize();
  }

  static uint64_t GetStateSize() {
    uint64_t state_size =
        decltype(fuse_conv1_relu6_bn_)::element_type::GetStateSize() +
        // decltype(fuse_conv2_relu6_bn_)::element_type::GetStateSize() +
        decltype(pad_)::element_type::GetStateSize() +
        decltype(conv2_)::element_type::GetStateSize() +
        decltype(relu6_)::element_type::GetStateSize() +
        decltype(bn_)::element_type::GetStateSize() +
        decltype(permute_)::element_type::GetStateSize() +
        decltype(fuse_bn1d_fc_)::element_type::GetStateSize();
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

    op_map(get_param, fuse_conv1_relu6_bn_, pad_, conv2_, relu6_, bn_, permute_,
           fuse_bn1d_fc_);
    return params;
  }
  LSTMNet_7_75 &GetLstm() { return lstms_; }
  float *GetPermuteOutput() {
    float *permute_output = reinterpret_cast<float *>(permute_->GetOutput());
    return permute_output;
  }

private:
  static constexpr uint64_t kInputSize =
      kBatchSize * 1 * kInputHeight * kInputWidth;
  static constexpr uint64_t kLength_Conv1 =
      (kInputWidth + 2 * 20 - 41) / 2 + 1; // 150
  static constexpr uint64_t kLength_Conv2 =
      (kLength_Conv1 + 2 * 10 - 21) / 2 + 1; // 75
  static constexpr uint64_t kOutputSize =
      kBatchSize * kLength_Conv2 * kNumClass;
  using Parent = Op<kOutputSize>;

  std::unique_ptr<Conv<kBatchSize, kInputHeight, kInputWidth, 1, 32, 11, 41, 5,
                       20, 2, 2, false, true, false, 1, true>>
      fuse_conv1_relu6_bn_; // [1,1,171,300],11,41 -> [1,32,86,150]
  std::unique_ptr<Padding<1, 32, 86, 150, 5, 10>> pad_;
  std::unique_ptr<Conv<kBatchSize, 96, 170, 32, 32, 11, 21, 0, 0, 1, 2, false,
                       false, false, 1, false>>
      conv2_;
  std::unique_ptr<ReLU<1, 32, 86, 75, 0, true>> relu6_;
  std::unique_ptr<BatchNormalization<1, 32, 86, 75>> bn_;

  std::unique_ptr<Permute3<kBatchSize, 2752, kLength_Conv2, 2, 0, 1>> permute_;

  LSTMNet_7_75 lstms_;

  float *lstm_output;

  std::unique_ptr<
      Linear<kBatchSize * kLength_Conv2, 256, kNumClass, true, false, true>>
      fuse_bn1d_fc_;
};
} // namespace mica::experiments::lstm
