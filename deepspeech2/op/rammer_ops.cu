#include "rammer_ops.h"

// Description:	Convolution
// Input:
//	- name: input0	type: float	shape: Shape{1, 32, 96, 170}
//	- name: input1	type: float	shape: Shape{32, 32, 11, 21}
// Output:
//	- name: Convolution_38_0	type: float	shape: Shape{1, 32, 86,
// 75}
// Launch:
// - dim3(43, 3, 1), dim3(2, 5, 32)
__device__ void DeviceDeepSpeechConv2(float *input0, float *input1,
                                      float *output0) {
  __shared__ float pad_temp_shared[138];
  __shared__ float input1_shared[672];
  {
    float *compute = output0;
    {
      float compute_local[5];

#pragma unroll
      for (int yy_c_init = 0; yy_c_init < 5; ++yy_c_init) {
        compute_local[yy_c_init] = 0.000000e+00f;
      }
#pragma unroll
      for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
#pragma unroll
        for (int rx_outer = 0; rx_outer < 11; ++rx_outer) {
          // __syncthreads();
          if ((threadIdx.z * 5 + threadIdx.y) < 138) {
            if (threadIdx.x == 0) {
              pad_temp_shared[(((((int)threadIdx.z) * 5) + ((int)threadIdx.x)) +
                               ((int)threadIdx.y))] =
                  input0[(
                      (((((rc_outer * 16320) + (((int)blockIdx.y) * 50)) +
                         (((((((int)threadIdx.z) * 5) + ((int)threadIdx.x)) +
                            ((int)threadIdx.y)) >>
                           1) *
                          1)) +
                        (((int)blockIdx.x) * 2 * 170)) +
                       rx_outer * 170) +
                      ((((((int)threadIdx.z) * 5) + ((int)threadIdx.x)) +
                        ((int)threadIdx.y)) &
                       1) *
                          170)];
            }
          }
#pragma unroll
          for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0;
               ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 3;
               ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
            if ((((((int)threadIdx.y) * 5) + (((int)threadIdx.x) * 3)) +
                 ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 21) {
              if (((((int)threadIdx.x) * 3) +
                   ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 5) {
                input1_shared[(
                    (((((int)threadIdx.z) * 21) + (((int)threadIdx.y) * 5)) +
                     (((int)threadIdx.x) * 3)) +
                    ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] =
                    input1[(
                        (((((((int)threadIdx.z) * 7392) + (rc_outer * 231)) +
                           (((int)threadIdx.y) * 5)) +
                          (((int)threadIdx.x) * 3)) +
                         (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner *
                          1)) +
                        rx_outer * 21)];
              }
            }
          }
          __syncthreads();
#pragma unroll
          for (int ry_inner = 0; ry_inner < 21; ++ry_inner) {
            float input1_temp = input1_shared[threadIdx.z * 21 + ry_inner];
#pragma unroll
            for (int yy_c = 0; yy_c < 5; ++yy_c) {
              compute_local[yy_c] +=
                  (pad_temp_shared[((((((int)threadIdx.y) * 20) + (yy_c * 4)) +
                                     (ry_inner * 2)) +
                                    ((int)threadIdx.x))] *
                   input1_temp);
            }
          }
        }
      }
#pragma unroll
      for (int yy_inner_inner_inner = 0; yy_inner_inner_inner < 5;
           ++yy_inner_inner_inner) {
        compute[((((((((int)threadIdx.z) * 6450) + (((int)blockIdx.y) * 25)) +
                    (((int)threadIdx.y) * 5)) +
                   (yy_inner_inner_inner * 1)) +
                  (((int)blockIdx.x) * 2 * 75)) +
                 ((int)threadIdx.x * 75))] =
            compute_local[yy_inner_inner_inner];
      }
    }
  }
}

__global__ void DeepSpeechConv2(const DeepSpeechConv2Input *__restrict__ input,
                                DeepSpeechConv2State *__restrict__ state) {
  DeviceDeepSpeechConv2((float *)input, (float *)&state->weight,
                        (float *)&state->output);
}

// 32, 86, 150 -> 32, 96, 170 ; 32 x 10, 1 x 1 x 1; 1, 86, 15
__global__ void RammerPadding(const PaddingInput *__restrict__ input,
                              PaddingState *__restrict__ state) {

  int input_offset =
      threadIdx.x * 86 * 150 + blockIdx.x * 150 + blockIdx.y * 10 + threadIdx.y;
  int output_offset = threadIdx.x * 96 * 170 + (blockIdx.x + 5) * 170 +
                      (blockIdx.y * 10 + threadIdx.y + 10);
  state->output.data[output_offset] = input->data.data[input_offset];
}