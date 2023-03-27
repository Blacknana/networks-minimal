#pragma once

#include <cfloat>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

const int kBlockSize = 256;
const int kTileSize = 16;

template <int kSize> __device__ struct Tensor {
  float data[kSize];
};

template <int kChannel> struct BatchNormParam {
  Tensor<kChannel> weight;
  Tensor<kChannel> bias;
  Tensor<kChannel> running_mean;
  Tensor<kChannel> running_var;
};

template <int kSize, int kChannel> struct BatchNormState {
  Tensor<kSize> output;
  BatchNormParam<kChannel> param;
};

template <int kBatchSize, int kInSize, int kOutSize, bool kIsBias,
          bool kFuseBatchNorm1d>
struct LinearState {
  Tensor<kBatchSize * kOutSize> output;
  Tensor<kInSize * kOutSize> weight;
  Tensor<kOutSize> bias;
  BatchNormParam<kInSize> bn_param;
};

template <int kBatchSize, int kInSize, int kOutSize>
struct LinearState<kBatchSize, kInSize, kOutSize, false, false> {
  Tensor<kBatchSize * kOutSize> output;
  Tensor<kInSize * kOutSize> weight;
};

template <int kBatchSize, int kInSize, int kOutSize>
struct LinearState<kBatchSize, kInSize, kOutSize, true, false> {
  Tensor<kBatchSize * kOutSize> output;
  Tensor<kInSize * kOutSize> weight;
  Tensor<kOutSize> bias;
};

template <int kOutputSize, int kChannel, int kFilterSize, int kColSize,
          int kBiasSize, bool kIsBias, bool kFuseBN>
struct ConvState;

template <int kOutputSize, int kChannel, int kFilterSize, int kColSize,
          int kBiasSize>
struct ConvState<kOutputSize, kChannel, kFilterSize, kColSize, kBiasSize, true,
                 false> {
  Tensor<kOutputSize> output;
  Tensor<kFilterSize> filter;
  Tensor<kBiasSize> bias;
  Tensor<kColSize> col;
};

template <int kOutputSize, int kChannel, int kFilterSize, int kColSize>
struct ConvState<kOutputSize, kChannel, kFilterSize, kColSize, 0, false,
                 false> {
  Tensor<kOutputSize> output;
  Tensor<kFilterSize> filter;
  Tensor<kColSize> col;
};

template <int kOutputSize, int kChannel, int kFilterSize, int kColSize>
struct ConvState<kOutputSize, kChannel, kFilterSize, kColSize, 0, false, true> {
  Tensor<kOutputSize> output;
  Tensor<kFilterSize> filter;
  BatchNormParam<kChannel> bn_param;
  Tensor<kColSize> col;
};

template <int kOutputSize, int kChannel, int kFilterSize, int kColSize,
          int kBiasSize>
struct ConvState<kOutputSize, kChannel, kFilterSize, kColSize, kBiasSize, true,
                 true> {
  Tensor<kOutputSize> output;
  Tensor<kFilterSize> filter;
  Tensor<kBiasSize> bias;
  BatchNormParam<kChannel> bn_param;
  Tensor<kColSize> col;
};

template <int kBatchSize, int kChannel, int kHeight, int kWidth, int kSize>
__global__ void operator_batch_normalization_h(
    const Tensor<kSize> *__restrict__ input,
    BatchNormState<kSize, kChannel> *__restrict__ state);

template <int kBatchSize, int kHeight, int kWidth, int kChannelIn,
          int kChannelOut, int kKernelH, int kKernelW, int kPadH, int kPadW,
          int kStrideH, int kStrideW, bool kIsBias, int kInputSize,
          int kColSize>
__global__ void im2col_h(const Tensor<kInputSize> *__restrict__ input,
                         Tensor<kColSize> *__restrict__ col);

template <int kSize, bool kReLU6>
__global__ void operator_vectorrelu_h(const Tensor<kSize> *input,
                                      Tensor<kSize> *output);

template <int kHeight, int kK, int kWidth, int kBroadCast, int kBatch1,
          int kBatch2, int kConvChannel, int kConvHeight, int kConvWidth,
          int kGroupSize = 1>
__global__ void operator_fuse_conv_relu6_bn_h(
    const Tensor<kBatch1 * kHeight * kK * kGroupSize> *__restrict__ input1,
    const Tensor<kBatch2 * kWidth * kK * kGroupSize> *__restrict__ input2,
    Tensor<kBatch1 * kHeight * kWidth * kBatch2 * kGroupSize>
        *__restrict__ output,
    const BatchNormParam<kConvChannel> *__restrict__ bn_param);

template <int kBatchSize, int kInSize, int kOutSize>
__global__ void operator_fuse_bn1d_linear_bias_h(
    const Tensor<kBatchSize * kInSize> *__restrict__ input1,
    const Tensor<kInSize * kOutSize> *__restrict__ input2,
    const Tensor<kOutSize> *__restrict__ bias,
    const BatchNormParam<kInSize> *__restrict__ bn_param,
    Tensor<kBatchSize * kOutSize> *__restrict__ output);

template <int kInSize0, int kInSize1, int kInSize2, int kOutSize0,
          int kOutSize1, int kOutSize2>
__global__ void operator_permute3_h(
    const Tensor<kInSize0 * kInSize1 * kInSize2> *__restrict__ input,
    Tensor<kInSize0 * kInSize1 * kInSize2> *__restrict__ output);