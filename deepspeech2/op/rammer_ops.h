#pragma once

#include "cuda_ops.h"
#include <cfloat>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

enum {
  kDeepSpeechConv2ChannelIn = 32,
  kDeepSpeechConv2ChannelOut = 32,
  kDeepSpeechConv2InputH = 96,
  kDeepSpeechConv2InputW = 170,
  kDeepSpeechConv2OutputH = 86,
  kDeepSpeechConv2OutputW = 75,
  kDeepSpeechConv2FilterH = 11,
  kDeepSpeechConv2FilterW = 21,
};

#pragma pack(push, 1)
struct DeepSpeechConv2Input {
  Tensor<kDeepSpeechConv2InputH * kDeepSpeechConv2ChannelIn *
         kDeepSpeechConv2InputW>
      data;
};

struct DeepSpeechConv2State {
  Tensor<kDeepSpeechConv2FilterH * kDeepSpeechConv2FilterW *
         kDeepSpeechConv2ChannelIn * kDeepSpeechConv2ChannelOut>
      weight;
  Tensor<kDeepSpeechConv2OutputH * kDeepSpeechConv2OutputW *
         kDeepSpeechConv2ChannelOut>
      output;
};
#pragma pack(pop)

__global__ void DeepSpeechConv2(const DeepSpeechConv2Input *__restrict__ input,
                                DeepSpeechConv2State *__restrict__ state);

struct PaddingInput {
  Tensor<412800> data;
};

struct PaddingState {
  Tensor<522240> output;
};

__global__ void RammerPadding(const PaddingInput *__restrict__ input,
                              PaddingState *__restrict__ state);