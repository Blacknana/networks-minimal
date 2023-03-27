#pragma once

#include "cuda_ops.h"
#include <memory>
#include <string_view>
#include <vector>

template <uint64_t kOutputSize> class Op {
public:
  Op() {}
  template <uint64_t T> void Forward(Tensor<T>) {
    throw std::runtime_error("not implement error");
  };

  virtual std::vector<std::string_view> GetParameters() = 0;

  static uint64_t GetStateSize() {
    throw std::runtime_error("not implement error");
    return 0;
  };

  virtual Tensor<kOutputSize> *GetOutput() { return output_; }

protected:
  Tensor<kOutputSize> *output_;

private:
  Op(const Op &other) = delete;
  Op(Op &&other) = delete;
  Op &operator=(const Op &other) = delete;
  Op &operator=(Op &&other) = delete;
};
