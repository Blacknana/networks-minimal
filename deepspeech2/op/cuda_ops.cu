#include "cuda_ops.h"

template <int kChannel>
static inline __device__ float batch_normalization_1d_func(
    float input, const BatchNormParam<kChannel> *__restrict__ bn_params,
    int index) {
  int channel = index % kChannel;
  float inv_var = bn_params->running_var.data[channel];
  inv_var = sqrtf(inv_var + 1e-5);
  inv_var = 1 / inv_var;
  float weight_v = bn_params->weight.data[channel];
  float bias_v = bn_params->bias.data[channel];
  float alpha = inv_var * weight_v;
  float mean_data = bn_params->running_mean.data[channel];
  float beta = bias_v - mean_data * alpha;
  return input * alpha + beta;
}

template <int kBatchSize, int kInSize, int kOutSize>
__global__ void operator_fuse_bn1d_linear_bias_h(
    const Tensor<kBatchSize * kInSize> *__restrict__ input1,
    const Tensor<kInSize * kOutSize> *__restrict__ input2,
    const Tensor<kOutSize> *__restrict__ bias,
    const BatchNormParam<kInSize> *__restrict__ bn_param,
    Tensor<kBatchSize * kOutSize> *__restrict__ output) {

  __shared__ float shared_input1[kTileSize][kTileSize];
  __shared__ float shared_input2[kTileSize][kTileSize];

  int bx = blockIdx.y;
  int by = blockIdx.x;
  int tx = threadIdx.y;
  int ty = threadIdx.x;

  int row = bx * kTileSize + tx;
  int col = by * kTileSize + ty;
  float v = 0;
  // #pragma unroll 8
  for (int i = 0; i < (kInSize + kTileSize - 1) / kTileSize; i++) {
    if (i * kTileSize + ty < kInSize && row < kBatchSize) {
      int index = row * kInSize + i * kTileSize + ty;
      float v1 = input1->data[index];
      shared_input1[tx][ty] =
          batch_normalization_1d_func<kInSize>(v1, bn_param, index);
    } else {
      shared_input1[tx][ty] = 0;
    }

    if (i * kTileSize + tx < kInSize && col < kOutSize)
      shared_input2[tx][ty] =
          input2->data[(i * kTileSize + tx) * kOutSize + col];
    else
      shared_input2[tx][ty] = 0;
    __syncthreads();
#pragma unroll 16
    for (int j = 0; j < kTileSize; j++)
      v += shared_input1[tx][j] * shared_input2[j][ty];
    __syncthreads();
  }

  if (row < kBatchSize && col < kOutSize) {
    v += bias->data[col];
    output->data[row * kOutSize + col] = v;
  }
}

template <int kBatchSize, int kChannel, int kHeight, int kWidth>
static inline __device__ float
batch_normalization_func(float input,
                         const BatchNormParam<kChannel> *__restrict__ bn_params,
                         int index) {
  int channel_size = kHeight * kWidth;
  int feature_size = kChannel * kHeight * kWidth;

  int batch = index / feature_size;
  int channel = (index - batch * kChannel * kHeight * kWidth) / channel_size;

  float inv_var = bn_params->running_var.data[channel];
  inv_var = sqrtf(inv_var + 1e-5);
  inv_var = 1 / inv_var;
  float weight_v = bn_params->weight.data[channel];
  float bias_v = bn_params->bias.data[channel];
  float alpha = inv_var * weight_v;
  float mean_data = bn_params->running_mean.data[channel];
  float beta = bias_v - mean_data * alpha;
  return input * alpha + beta;
}

template <int kBatchSize, int kChannel, int kHeight, int kWidth, int kSize>
__global__ void operator_batch_normalization_h(
    const Tensor<kSize> *__restrict__ input,
    BatchNormState<kSize, kChannel> *__restrict__ state) {
  int total_size = kBatchSize * kChannel * kHeight * kWidth;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < total_size) {
    state->output.data[tid] =
        batch_normalization_func<kBatchSize, kChannel, kHeight, kWidth>(
            input->data[tid], &state->param, tid);
  }
}

template <int kHeight, int kK, int kWidth, int kBroadCast, int kBatch1,
          int kBatch2, int kConvChannel, int kConvHeight, int kConvWidth,
          int kGroupSize>
__global__ void operator_fuse_conv_relu6_bn_h(
    const Tensor<kBatch1 * kHeight * kK * kGroupSize> *__restrict__ input1,
    const Tensor<kBatch2 * kWidth * kK * kGroupSize> *__restrict__ input2,
    Tensor<kBatch1 * kHeight * kWidth * kBatch2 * kGroupSize>
        *__restrict__ output,
    const BatchNormParam<kConvChannel> *__restrict__ bn_param) {

  static_assert(kConvChannel * kConvHeight * kConvWidth ==
                kWidth * kHeight * kGroupSize);

  __shared__ float shared_input1[kTileSize][kTileSize];
  __shared__ float shared_input2[kTileSize][kTileSize];

  int block_z = blockIdx.z;
  int batch_idx = block_z / kGroupSize;
  int group_idx = block_z % kGroupSize;
  int input1_off = 0;
  int input2_off = 0;
  int output_off = 0;
  if (kBroadCast != 1)
    input1_off = batch_idx * kHeight * kK * kGroupSize;
  if (kBroadCast != 2)
    input2_off = batch_idx * kK * kWidth * kGroupSize;
  input1_off += group_idx * kHeight * kK;
  input2_off += group_idx * kK * kWidth;

  output_off =
      batch_idx * kGroupSize * kHeight * kWidth + group_idx * kHeight * kWidth;

  int bx = blockIdx.y;
  int by = blockIdx.x;
  int tx = threadIdx.y;
  int ty = threadIdx.x;

  int row = bx * kTileSize + tx;
  int col = by * kTileSize + ty;
  float v = 0;
  // #pragma unroll 64
  for (int i = 0; i < (kK + kTileSize - 1) / kTileSize; i++) {
    if (i * kTileSize + ty < kK && row < kHeight)
      shared_input1[tx][ty] =
          input1->data[input1_off + row * kK + i * kTileSize + ty];
    else
      shared_input1[tx][ty] = 0;

    if (i * kTileSize + tx < kK && col < kWidth)
      shared_input2[tx][ty] =
          input2->data[input2_off + (i * kTileSize + tx) * kWidth + col];
    else
      shared_input2[tx][ty] = 0;
    __syncthreads();
#pragma unroll 16
    for (int j = 0; j < kTileSize; j++)
      v += shared_input1[tx][j] * shared_input2[j][ty];
    __syncthreads();
  }

  if (row < kHeight && col < kWidth) {
    v = v > 0 ? v : 0;
    v = v < 6 ? v : 6;
    int index = output_off + row * kWidth + col;
    output->data[index] =
        batch_normalization_func<kBatch2, kConvChannel, kConvHeight,
                                 kConvWidth>(v, bn_param, index);
  }
}

template <int kBatchSize, int kHeight, int kWidth, int kChannelIn,
          int kChannelOut, int kKernelH, int kKernelW, int kPadH, int kPadW,
          int kStrideH, int kStrideW, bool kIsBias, int kInputSize,
          int kColSize>
__global__ void im2col_h(const Tensor<kInputSize> *__restrict__ input,
                         Tensor<kColSize> *__restrict__ col) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  int height_col = (kHeight + 2 * kPadH - kKernelH) / kStrideH + 1;
  int width_col = (kWidth + 2 * kPadW - kKernelW) / kStrideW + 1;
  int n = kChannelIn * height_col * width_col;
  int im_stride = kChannelIn * kHeight * kWidth;
  int col_stride = kChannelIn * kKernelH * kKernelW * height_col * width_col;

  if (index < n) {
    int batch_idx = blockIdx.y;
    int input_offset = batch_idx * im_stride;
    int col_offset = batch_idx * col_stride;

    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kKernelH * kKernelW;
    const int h_offset = h_col * kStrideH - kPadH;
    const int w_offset = w_col * kStrideW - kPadW;

    // channel offset
    col_offset += (c_col * height_col + h_col) * width_col + w_col;
    input_offset += (c_im * kHeight + h_offset) * kWidth + w_offset;

// copy to col
#pragma unroll 4
    for (int i = 0; i < kKernelH; ++i) {
#pragma unroll 4
      for (int j = 0; j < kKernelW; ++j) {
        int h_im = h_offset + i;
        int w_im = w_offset + j;
        col->data[col_offset] =
            (h_im >= 0 && w_im >= 0 && h_im < kHeight && w_im < kWidth)
                ? input->data[input_offset + i * kWidth + j]
                : 0;
        col_offset += height_col * width_col;
      }
    }
  }
}

template <int kSize, bool kReLU6>
__global__ void operator_vectorrelu_h(const Tensor<kSize> *input,
                                      Tensor<kSize> *output) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < kSize) {
    float v = input->data[tid];
    v = v > 0 ? v : 0;
    if (kReLU6) {
      v < 6 ? v : 6;
    }
    output->data[tid] = v;
  }
}

template <int kSize0, int kSize1, int kSize2, int kStride0, int kStride1,
          int kStride2>
__global__ void
operator_permute3_h(const Tensor<kSize0 * kSize1 * kSize2> *__restrict__ input,
                    Tensor<kSize0 * kSize1 * kSize2> *__restrict__ output) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < kSize0 * kSize1 * kSize2) {
    int z_idx = idx % kSize2;
    idx /= kSize2;
    int y_idx = idx % kSize1;
    int x_idx = idx / kSize1;

    int input_off = x_idx * kStride0 + y_idx * kStride1 + z_idx * kStride2;
    int output_off = x_idx * kSize1 * kSize2 + y_idx * kSize2 + z_idx;

    output->data[output_off] = input->data[input_off];
  }
}

template __global__ void
im2col_h<1, 171, 300, 1, 32, 11, 41, 5, 20, 2, 2, false, 51300, 5817900>(
    Tensor<51300> const *, Tensor<5817900> *);
template __global__ void
operator_fuse_conv_relu6_bn_h<32, 451, 12900, 1, 1, 1, 32, 86, 150, 1>(
    Tensor<(((1) * (32)) * (451)) * (1)> const *,
    Tensor<(((1) * (12900)) * (451)) * (1)> const *,
    Tensor<((((1) * (32)) * (12900)) * (1)) * (1)> *,
    BatchNormParam<32> const *);
template __global__ void operator_permute3_h<75, 1, 2752, 1, 206400, 75>(
    Tensor<((75) * (1)) * (2752)> const *, Tensor<((75) * (1)) * (2752)> *);
template __global__ void operator_fuse_bn1d_linear_bias_h<75, 256, 29>(
    Tensor<(75) * (256)> const *, Tensor<(29) * (256)> const *,
    Tensor<29> const *, BatchNormParam<256> const *, Tensor<(75) * (29)> *);
template __global__ void
operator_vectorrelu_h<206400, true>(Tensor<206400> const *, Tensor<206400> *);
template __global__ void operator_batch_normalization_h<1, 32, 86, 75, 206400>(
    Tensor<206400> const *, BatchNormState<206400, 32> *);