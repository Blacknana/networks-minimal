#pragma once

#include "cuda_ops.h"
#include "op.h"
#include "utils.h"

template <int kSize0, int kSize1, int kSize2, int kDim0, int kDim1, int kDim2>
class Permute3 : public Op<kSize0 * kSize1 * kSize2> {
public:
  Permute3(void *device_ptr) {
    Parent::output_ = reinterpret_cast<Tensor<kSize> *>(device_ptr);
  };

  template <int T> void Forward(Tensor<T> *input) {
    static_assert(T == kSize);

    static constexpr int input_size[] = {kSize0, kSize1, kSize2};
    static constexpr int input_stride[] = {kSize1 * kSize2, kSize2, 1};
    static constexpr int kOutputSize0 = input_size[kDim0],
                         kOutputSize1 = input_size[kDim1],
                         kOutputSize2 = input_size[kDim2];
    static constexpr int kOutputStride0 = input_stride[kDim0],
                         kOutputStride1 = input_stride[kDim1],
                         kOutputStride2 = input_stride[kDim2];

    uint64_t grid_size = (T + kBlockSize - 1) / kBlockSize;
    void *output = Parent::output_;
    void *args[] = {&input, &output};
    cudaLaunchKernel(
        (const void *)
            operator_permute3_h<kOutputSize0, kOutputSize1, kOutputSize2,
                                kOutputStride0, kOutputStride1, kOutputStride2>,
        dim3(grid_size), dim3(kBlockSize), (void **)args, 0);
    CUDA_POST_KERNEL_CHECK;
  };

  static uint64_t GetStateSize() { return kSize; };

  virtual std::vector<std::string_view> GetParameters() override { return {}; }

private:
  static constexpr uint64_t kSize = kSize0 * kSize1 * kSize2;
  using Parent = Op<kSize>;
};
