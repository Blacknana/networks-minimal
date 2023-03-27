#pragma once

#include "op.h"
#include "utils.h"

template <int kBatchSize, int kChannels, int kHeight, int kWidth>
class Flatten : public Op<kBatchSize * kChannels * kHeight * kWidth> {
public:
  Flatten(void *device_ptr) {}
  static uint64_t GetStateSize() { return 0; };

  template <int T> void Forward(Tensor<T> *input) {
    static_assert(T == kSize);
    Parent::output_ = input;
  };

  virtual std::vector<std::string_view> GetParameters() override { return {}; }

private:
  static constexpr uint64_t kSize = kBatchSize * kChannels * kHeight * kWidth;
  using Parent = Op<kSize>;
};
