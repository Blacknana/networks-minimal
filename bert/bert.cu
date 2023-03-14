#include "gtest/gtest.h"
#include <absl/types/span.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>

template <class T>
static CUresult LaunchKernel(CUfunction f, unsigned grid_x, unsigned block_x,
                             CUstream stream, const T &param,
                             unsigned shared = 0) {
  size_t size = sizeof(T);
  void *config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, const_cast<T *>(&param),
                    CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
  return cuLaunchKernel(f, grid_x, 1, 1, block_x, 1, 1, shared, stream, nullptr,
                        config);
}

#define CU_CHECK(error)                                                        \
  {                                                                            \
    if (error != CUDA_SUCCESS) {                                               \
      const char *error_name;                                                  \
      cuGetErrorName(error, &error_name);                                      \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error, error_name);            \
      exit(1);                                                                 \
    }                                                                          \
  }

#define CUDA_CHECK(error)                                                      \
  {                                                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

namespace bert {

enum BertScaleParams {
  kBatchSize = 1,
  kSeqLength = 384,
  kHeadNum = 12,
  kHeadSize = 64,
  kLayerNum = 12,
  kHiddenSize = 4,
  kHiddenDim = kHeadNum * kHeadSize,
};

enum BertGemmParams {
  kWmmaM = 16,
  kWmmaN = 16,
  kWmmaK = 16,
  kChunkK = 4,
  kStage = 3,
  kBlockRowWarps = 2,
  kBlockColWarps = 2,
  kWarpSize = 32,
  kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
  kInputSkew = 8,
  kAccSkew = 8,

  kGemmK1WarpRowTiles = 1,
  kGemmK1WarpColTiles = 3,
  kGemmK1BatchedNum = 1,

  kGemmK2WarpRowTiles = 4,
  kGemmK2WarpColTiles = 4,
  kGemmK2BatchedNum = kHeadNum,

  kGemmK3WarpRowTiles = 2,
  kGemmK3WarpColTiles = 2,
  kGemmK3BatchedNum = kHeadNum,

  kGemmK4WarpRowTiles = 2,
  kGemmK4WarpColTiles = 2,

  kGemmK5WarpRowTiles = 4,
  kGemmK5WarpColTiles = 3,

  kGemmK6BlockRowTiles = 4,
  kGemmK6BlockColTiles = 3,
  kGemmK6BlockSliceKTiles = 4,
};

struct BertWordVec {
  half data[kHiddenDim];
};

struct BertWeight {
  half data[kHiddenDim][kHiddenDim];
};

struct BertAttrMask {
  half data[kBatchSize][kSeqLength][kSeqLength];
};

#pragma pack(push, 1)
struct BertInput {
  BertWordVec words[kBatchSize][kSeqLength];
};
#pragma pack(pop)

} // namespace bert

using namespace bert;

#pragma pack(push, 1)
struct GemmParam {
  CUdeviceptr matrix_a;
  CUdeviceptr matrix_b;
  CUdeviceptr matrix_c;
};

struct AddBiasActParam {
  CUdeviceptr out;
  CUdeviceptr bias;
};

struct AddBiasInputLayernormParam {
  CUdeviceptr out;
  CUdeviceptr input;
  CUdeviceptr bias;
  CUdeviceptr gamma;
  CUdeviceptr beta;
};

struct GemmAddBiasParam {
  CUdeviceptr matrix_a;
  CUdeviceptr matrix_b;
  CUdeviceptr bias;
  CUdeviceptr matrix_c;
};

struct SoftmaxParam {
  CUdeviceptr qk_buf;
  CUdeviceptr attr_mask;
  half scalar;
};
#pragma pack(pop)

struct AttentionWeight {
  CUdeviceptr d_query_weight;
  CUdeviceptr d_query_bias;
  CUdeviceptr d_key_weight;
  CUdeviceptr d_key_bias;
  CUdeviceptr d_value_weight;
  CUdeviceptr d_value_bias;
  CUdeviceptr d_output_weight;
  CUdeviceptr d_output_bias;
};

struct LayerNormWeight {
  CUdeviceptr d_gamma;
  CUdeviceptr d_beta;
};

struct FFNWeight {
  CUdeviceptr d_inter_weight;
  CUdeviceptr d_inter_bias;
  CUdeviceptr d_output_weight;
  CUdeviceptr d_output_bias;
};

struct AttentionParam {
  CUdeviceptr d_from_tensor;
  CUdeviceptr d_attr_mask;
  CUdeviceptr d_attr_out;
  AttentionWeight d_self_attention;
};

struct BertParam {
  CUdeviceptr d_from_tensor;
  CUdeviceptr d_attr_mask;
  CUdeviceptr d_transformer_out;
  AttentionWeight d_self_attention;
  LayerNormWeight d_self_layernorm;
  FFNWeight d_ffn;
  LayerNormWeight d_ffn_layernorm;
};

class Attention {
public:
  void Initialize(AttentionParam param, int max_shm_per_block);
  void Solve();
  void Finalize();

private:
  AttentionParam param_;
  CUfunction cu_gemm_add_qkv_bias_;
  CUfunction cu_gemm_k2_;
  CUfunction cu_gemm_reshape_;
  CUfunction cu_softmax_;
  CUdeviceptr d_query_buf_;
  CUdeviceptr d_key_buf_;
  CUdeviceptr d_value_buf_;
  CUdeviceptr d_qk_buf_;
};

class Bert {
public:
  explicit Bert(absl::Span<const float> src_model);
  bool Initialize(absl::Span<const float> input);
  void Solve();
  bool Fetch(absl::Span<float> output);
  void Finalize();

private:
  BertParam param_;
  Attention attention_;
  int max_shm_per_block_;
  CUdevice cu_device_;
  CUcontext cu_context_;
  CUfunction cu_gemm_k4_;
  CUfunction cu_gemm_k5_;
  CUfunction cu_gemm_k6_;
  CUfunction cu_add_bias_input_layernorm_;
  CUfunction cu_add_bias_act_;
  CUdeviceptr d_attr_out_buf_;
  CUdeviceptr d_attr_matmul_buf_;
  CUdeviceptr d_inter_matmul_buf_;
  CUdeviceptr d_attr_matmul_unnormed_buf_;
};

__global__ void gemm_add_qkv_bias(const half *__restrict__ matrix_a,
                                  const half *__restrict__ matrix_b,
                                  const half *__restrict__ bias,
                                  half *__restrict__ matrix_c);
__global__ void gemm_k2(const half *__restrict__ matrix_a,
                        const half *__restrict__ matrix_b,
                        half *__restrict__ matrix_c);
__global__ void gemm_reshape(const half *__restrict__ matrix_a,
                             const half *__restrict__ matrix_b,
                             half *__restrict__ matrix_c);
template <int kWarpRowTiles, int kWarpColTiles, int M, int N, int K, int B>
__global__ void gemm_three_stage(const half *__restrict__ matrix_a,
                                 const half *__restrict__ matrix_b,
                                 half *__restrict__ matrix_c);
__global__ void gemm_k6(const half *__restrict__ matrix_a,
                        const half *__restrict__ matrix_b,
                        half *__restrict__ matrix_c);
__global__ void add_bias_input_layernorm(BertInput *out, const BertInput *input,
                                         const BertWordVec *bias,
                                         const BertWordVec *gamma,
                                         const BertWordVec *beta);
__global__ void add_bias_gelu(BertInput *out,
                              const BertWordVec *__restrict bias);
__global__ void softmax(half *qk_buf_, const half *attr_mask,
                        const half scalar);

std::vector<float> ReadFloatFromFile(const std::string &path) {
  std::ifstream in_file(path);
  in_file.setf(std::ios::fixed, std::ios::floatfield);
  return std::vector<float>(std::istream_iterator<float>(in_file),
                            std::istream_iterator<float>());
}

TEST(TestBERT, test_bert) {
  auto model = ReadFloatFromFile("bert_model_params.txt");
  auto input = ReadFloatFromFile("bert_input_params.txt");
  auto expect_result = ReadFloatFromFile("bert_expect_results.txt");
  std::vector<float> output_result_buffer(expect_result.size());
  absl::Span<float> output(output_result_buffer);
  auto network = new Bert(model);

  ASSERT_TRUE(network->Initialize(input));
  network->Solve();
  ASSERT_TRUE(network->Fetch(output));
  for (unsigned int i = 0; i < expect_result.size(); ++i) {
    ASSERT_NEAR(output[i], expect_result[i], 1e-5);
  }

  enum { kWarmUp = 200, kLoop = 1000 };
  // Warm-up, 100 times
  for (int i = 0; i < kWarmUp; i++) {
    network->Initialize(input);
    network->Solve();
    network->Fetch(output);
  }
  double min_ms = std::numeric_limits<double>::max();
  double max_ms = std::numeric_limits<double>::min();
  double total_ms = 0.00000f;
  for (int i = 0; i < kLoop; i++) {
    auto start = std::chrono::steady_clock::now();
    network->Initialize(input);
    network->Solve();
    network->Fetch(output);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;
    double iteration_ms = elapsed.count();
    printf("Iteration time %f us\n", iteration_ms);
    min_ms = std::min(iteration_ms, min_ms);
    max_ms = std::max(iteration_ms, max_ms);
    total_ms = total_ms + iteration_ms;
  }
  printf("Sumamry: [min, max, mean] = [%f, %f, %f] us\n", min_ms, max_ms,
         total_ms / kLoop);

  network->Finalize();
}

void Attention::Initialize(AttentionParam param, int max_shm_per_block) {
  CU_CHECK(cuMemAlloc(&d_query_buf_,
                      sizeof(BertInput) * 3 + sizeof(BertAttrMask) * kHeadNum));
  CUDA_CHECK(cudaGetFuncBySymbol(&cu_gemm_add_qkv_bias_,
                                 (const void *)gemm_add_qkv_bias));
  CUDA_CHECK(cudaGetFuncBySymbol(&cu_gemm_k2_, (const void *)gemm_k2));
  CUDA_CHECK(
      cudaGetFuncBySymbol(&cu_gemm_reshape_, (const void *)gemm_reshape));
  CUDA_CHECK(cudaGetFuncBySymbol(&cu_softmax_, (const void *)softmax));
  CU_CHECK(cuFuncSetAttribute(cu_gemm_add_qkv_bias_,
                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              max_shm_per_block));
  CU_CHECK(cuFuncSetAttribute(cu_gemm_reshape_,
                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              max_shm_per_block));

  d_key_buf_ = d_query_buf_ + sizeof(BertInput);
  d_value_buf_ = d_key_buf_ + sizeof(BertInput);
  d_qk_buf_ = d_value_buf_ + sizeof(BertInput);

  param_ = param;
}

void Attention::Solve() {
  const int m = kBatchSize * kSeqLength;
  const int k = kHiddenDim;
  const int n = k;
  const half scalar = 1 / sqrtf(kHeadSize * 1.0f);

  GemmAddBiasParam gemm_k1_params = {
      param_.d_self_attention.d_query_weight, param_.d_from_tensor,
      param_.d_self_attention.d_query_bias, d_query_buf_};
  const int gemm_k1_blocks =
      (n / (kBlockRowWarps * kGemmK1WarpRowTiles * kWmmaM)) *
      (m / (kBlockColWarps * kGemmK1WarpColTiles * kWmmaN)) * kGemmK1BatchedNum;
  const int gemm_k1_shared_mem =
      (kStage *
       (3 * kChunkK * kWmmaK *
            (kBlockRowWarps * kGemmK1WarpRowTiles * kWmmaM + kInputSkew) +
        kBlockColWarps * kGemmK1WarpColTiles * kWmmaN *
            (kChunkK * kWmmaK + kInputSkew))) *
      sizeof(half);
  CU_CHECK(LaunchKernel(cu_gemm_add_qkv_bias_, gemm_k1_blocks, kBlockThreads, 0,
                        gemm_k1_params, gemm_k1_shared_mem));

  GemmParam gemm_k2_params = {d_key_buf_, d_query_buf_, d_qk_buf_};
  const int gemm_k2_blocks =
      (m / (kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM)) *
      (m / (kBlockColWarps * kGemmK2WarpColTiles * kWmmaN)) * kGemmK2BatchedNum;
  const int gemm_k2_shared_mem =
      (kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM *
           (kChunkK * kWmmaK + kInputSkew) +
       kBlockColWarps * kGemmK2WarpColTiles * kWmmaN *
           (kChunkK * kWmmaK + kInputSkew)) *
      sizeof(half);
  CU_CHECK(LaunchKernel(cu_gemm_k2_, gemm_k2_blocks, kBlockThreads, 0,
                        gemm_k2_params, gemm_k2_shared_mem));

  SoftmaxParam softmax_params = {d_qk_buf_, param_.d_attr_mask, scalar};
  CU_CHECK(LaunchKernel(cu_softmax_, kBatchSize * kSeqLength * kHeadNum,
                        kWarpSize, 0, softmax_params));

  GemmParam gemm_k3_params = {d_value_buf_, d_qk_buf_, param_.d_attr_out};
  const int gemm_k3_blocks =
      (kHeadSize / (kBlockRowWarps * kGemmK3WarpRowTiles * kWmmaM)) *
      (m / (kBlockColWarps * kGemmK3WarpColTiles * kWmmaN)) * kGemmK3BatchedNum;
  const int gemm_k3_shared_mem =
      (kStage *
       (kChunkK * kWmmaK *
            (kBlockRowWarps * kGemmK3WarpRowTiles * kWmmaM + kInputSkew) +
        kBlockColWarps * kGemmK3WarpColTiles * kWmmaN *
            (kChunkK * kWmmaK + kInputSkew))) *
      sizeof(half);
  CU_CHECK(LaunchKernel(cu_gemm_reshape_, gemm_k3_blocks, kBlockThreads, 0,
                        gemm_k3_params, gemm_k3_shared_mem));
}

void Attention::Finalize() { CU_CHECK(cuMemFree(d_query_buf_)); }

Bert::Bert(absl::Span<const float> src_model) {
  CU_CHECK(cuInit(0));
  CU_CHECK(cuDeviceGet(&cu_device_, 0));
  CU_CHECK(cuDeviceGetAttribute(
      &max_shm_per_block_,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cu_device_));
  CU_CHECK(cuCtxCreate(&cu_context_, 0, cu_device_));
  CUDA_CHECK(cudaGetFuncBySymbol(&cu_add_bias_input_layernorm_,
                                 (const void *)add_bias_input_layernorm));
  CUDA_CHECK(
      cudaGetFuncBySymbol(&cu_add_bias_act_, (const void *)add_bias_gelu));
  CUDA_CHECK(cudaGetFuncBySymbol(
      &cu_gemm_k4_,
      (const void *)gemm_three_stage<kGemmK4WarpRowTiles, kGemmK4WarpColTiles,
                                     kHiddenDim, kSeqLength, kHiddenDim, 1>));
  CUDA_CHECK(cudaGetFuncBySymbol(
      &cu_gemm_k5_,
      (const void *)gemm_three_stage<kGemmK5WarpRowTiles, kGemmK5WarpColTiles,
                                     kHiddenSize * kHiddenDim, kSeqLength,
                                     kHiddenDim, 1>));
  CUDA_CHECK(cudaGetFuncBySymbol(&cu_gemm_k6_, (const void *)gemm_k6));
  CU_CHECK(cuFuncSetAttribute(cu_gemm_k4_,
                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              max_shm_per_block_));
  CU_CHECK(cuFuncSetAttribute(cu_gemm_k5_,
                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              max_shm_per_block_));
  CU_CHECK(cuFuncSetAttribute(cu_gemm_k6_,
                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              max_shm_per_block_));

  CU_CHECK(cuMemAlloc(&param_.d_from_tensor, sizeof(BertInput)));
  param_.d_transformer_out = param_.d_from_tensor;

  size_t model_size =
      sizeof(BertAttrMask) + sizeof(BertWeight) * 12 + sizeof(BertWordVec) * 13;
  CU_CHECK(cuMemAlloc(&param_.d_attr_mask, model_size));
  CU_CHECK(cuMemcpyHtoD(param_.d_attr_mask, src_model.data(), model_size));

  param_.d_self_attention.d_query_weight =
      param_.d_attr_mask + sizeof(BertAttrMask);
  param_.d_self_attention.d_key_weight =
      param_.d_self_attention.d_query_weight + sizeof(BertWeight);
  param_.d_self_attention.d_value_weight =
      param_.d_self_attention.d_key_weight + sizeof(BertWeight);
  param_.d_self_attention.d_output_weight =
      param_.d_self_attention.d_value_weight + sizeof(BertWeight);
  param_.d_ffn.d_inter_weight =
      param_.d_self_attention.d_output_weight + sizeof(BertWeight);
  param_.d_ffn.d_output_weight =
      param_.d_ffn.d_inter_weight + sizeof(BertWeight) * kHiddenSize;
  param_.d_self_attention.d_query_bias =
      param_.d_ffn.d_output_weight + sizeof(BertWeight) * kHiddenSize;
  param_.d_self_attention.d_key_bias =
      param_.d_self_attention.d_query_bias + sizeof(BertWordVec);
  param_.d_self_attention.d_value_bias =
      param_.d_self_attention.d_key_bias + sizeof(BertWordVec);
  param_.d_self_attention.d_output_bias =
      param_.d_self_attention.d_value_bias + sizeof(BertWordVec);
  param_.d_self_layernorm.d_gamma =
      param_.d_self_attention.d_output_bias + sizeof(BertWordVec);
  param_.d_self_layernorm.d_beta =
      param_.d_self_layernorm.d_gamma + sizeof(BertWordVec);
  param_.d_ffn.d_inter_bias =
      param_.d_self_layernorm.d_beta + sizeof(BertWordVec);
  param_.d_ffn.d_output_bias =
      param_.d_ffn.d_inter_bias + sizeof(BertWordVec) * kHiddenSize;
  param_.d_ffn_layernorm.d_gamma =
      param_.d_ffn.d_output_bias + sizeof(BertWordVec);
  param_.d_ffn_layernorm.d_beta =
      param_.d_ffn_layernorm.d_gamma + sizeof(BertWordVec);

  CU_CHECK(cuMemAlloc(&d_attr_out_buf_, sizeof(BertInput) * 7));
  d_attr_matmul_buf_ = d_attr_out_buf_ + sizeof(BertInput);
  d_inter_matmul_buf_ = d_attr_matmul_buf_ + sizeof(BertInput);
  d_attr_matmul_unnormed_buf_ =
      d_inter_matmul_buf_ + sizeof(BertInput) * kHiddenSize;

  attention_.Initialize({param_.d_from_tensor, param_.d_attr_mask,
                         d_attr_out_buf_, param_.d_self_attention},
                        max_shm_per_block_);
}

bool Bert::Initialize(absl::Span<const float> input) {
  CU_CHECK(cuMemcpyHtoD(param_.d_from_tensor, input.data(), sizeof(BertInput)));
  return true;
}

void Bert::Solve() {
  for (int i = 0; i < kLayerNum; ++i) {
    attention_.Solve();

    int m = kBatchSize * kSeqLength;
    int k = kHiddenDim;
    int n = k;

    GemmParam gemm_k4_params = {param_.d_self_attention.d_output_weight,
                                d_attr_out_buf_, d_attr_matmul_buf_};
    const int gemm_k4_blocks =
        (n / (kBlockRowWarps * kGemmK4WarpRowTiles * kWmmaM)) *
        (m / (kBlockColWarps * kGemmK4WarpColTiles * kWmmaN));
    const int gemm_k4_shared_mem =
        (kStage *
         (kChunkK * kWmmaK *
              (kBlockRowWarps * kGemmK4WarpRowTiles * kWmmaM + kInputSkew) +
          kBlockColWarps * kGemmK4WarpColTiles * kWmmaN *
              (kChunkK * kWmmaK + kInputSkew))) *
        sizeof(half);
    CU_CHECK(LaunchKernel(cu_gemm_k4_, gemm_k4_blocks, kBlockThreads, 0,
                          gemm_k4_params, gemm_k4_shared_mem));

    AddBiasInputLayernormParam add_bias_input_layernorm_params = {
        d_attr_matmul_buf_, param_.d_from_tensor,
        param_.d_self_attention.d_output_bias, param_.d_self_layernorm.d_gamma,
        param_.d_self_layernorm.d_beta};
    CU_CHECK(LaunchKernel(cu_add_bias_input_layernorm_, m, n / 2, 0,
                          add_bias_input_layernorm_params));

    n *= kHiddenSize;

    GemmParam gemm_k5_params = {param_.d_ffn.d_inter_weight, d_attr_matmul_buf_,
                                d_inter_matmul_buf_};
    const int gemm_k5_blocks =
        (n / (kBlockRowWarps * kGemmK5WarpRowTiles * kWmmaM)) *
        (m / (kBlockColWarps * kGemmK5WarpColTiles * kWmmaN));
    const int gemm_k5_shared_mem =
        (kStage *
         (kChunkK * kWmmaK *
              (kBlockRowWarps * kGemmK5WarpRowTiles * kWmmaM + kInputSkew) +
          kBlockColWarps * kGemmK5WarpColTiles * kWmmaN *
              (kChunkK * kWmmaK + kInputSkew))) *
        sizeof(half);
    CU_CHECK(LaunchKernel(cu_gemm_k5_, gemm_k5_blocks, kBlockThreads, 0,
                          gemm_k5_params, gemm_k5_shared_mem));

    AddBiasActParam add_bias_act_params = {d_inter_matmul_buf_,
                                           param_.d_ffn.d_inter_bias};
    CU_CHECK(LaunchKernel(cu_add_bias_act_, m, n / 8, 0, add_bias_act_params));

    n = k;
    k *= kHiddenSize;

    GemmParam gemm_k6_params = {param_.d_ffn.d_output_weight,
                                d_inter_matmul_buf_, param_.d_transformer_out};
    const int gemm_k6_blocks = (n / (kGemmK6BlockRowTiles * kWmmaM)) *
                               (m / (kGemmK6BlockColTiles * kWmmaN));
    const int gemm_k6_shared_mem =
        (kStage * (kGemmK6BlockSliceKTiles * kWmmaK *
                       (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
                   kGemmK6BlockColTiles * kWmmaN *
                       (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew))) *
        sizeof(half);
    CU_CHECK(LaunchKernel(cu_gemm_k6_, gemm_k6_blocks, kBlockThreads, 0,
                          gemm_k6_params, gemm_k6_shared_mem));

    add_bias_input_layernorm_params = {
        param_.d_transformer_out, d_attr_matmul_buf_,
        param_.d_ffn.d_output_bias, param_.d_ffn_layernorm.d_gamma,
        param_.d_ffn_layernorm.d_beta};
    CU_CHECK(LaunchKernel(cu_add_bias_input_layernorm_, m, n / 2, 0,
                          add_bias_input_layernorm_params));
  }
}

bool Bert::Fetch(absl::Span<float> output) {
  CU_CHECK(
      cuMemcpyDtoH(output.data(), param_.d_transformer_out, sizeof(BertInput)));
  return true;
}

void Bert::Finalize() {
  attention_.Finalize();
  CU_CHECK(cuMemFree(param_.d_from_tensor));
  CU_CHECK(cuMemFree(param_.d_attr_mask));
  CU_CHECK(cuMemFree(d_attr_out_buf_));
  CU_CHECK(cuCtxDestroy(cu_context_));
}

__inline__ __device__ float warpReduceSum(float val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  return val;
}

__inline__ __device__ float blockReduceSum(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum(val);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
  val = warpReduceSum(val);
  return val;
}

__inline__ __device__ float warpReduceMax(float val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
  return val;
}

__inline__ __device__ int target_index(int id1, int id2, int id3, int id4,
                                       int dim_1, int dim_2, int dim_3,
                                       int dim_4) {
  return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 +
         id4;
}

__inline__ __device__ half2 gelu(half2 val) {
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp = __half22float2(val);

  tmp.x =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

__global__ void add_bias_input_layernorm(BertInput *out, const BertInput *input,
                                         const BertWordVec *bias,
                                         const BertWordVec *gamma,
                                         const BertWordVec *beta) {
  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  float2 local_out_fp2;

  half2 *out_ptr = reinterpret_cast<half2 *>(out);
  const half2 *input_ptr = reinterpret_cast<const half2 *>(input);
  const half2 *bias_ptr = reinterpret_cast<const half2 *>(bias);
  const half2 *gamma_ptr = reinterpret_cast<const half2 *>(gamma);
  const half2 *beta_ptr = reinterpret_cast<const half2 *>(beta);

  float local_out = 0.0f;
  int id = blockIdx.x * kHiddenDim / 2 + tid;
  local_out_fp2 = __half22float2(
      __hadd2(__hadd2(out_ptr[id], input_ptr[id]), __ldg(&bias_ptr[tid])));
  local_out += local_out_fp2.x;
  local_out += local_out_fp2.y;

  mean = blockReduceSum(local_out);
  if (threadIdx.x == 0)
    s_mean = mean / kHiddenDim;
  __syncthreads();

  variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
  variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
  variance = blockReduceSum(variance);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / kHiddenDim + 1e-6f);
  __syncthreads();

  float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
  float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
  local_out_fp2.x =
      (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
  local_out_fp2.y =
      (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
  out_ptr[id] = __float22half2_rn(local_out_fp2);
}

__global__ void add_bias_gelu(BertInput *out,
                              const BertWordVec *__restrict bias) {
  const int m = kBatchSize * kSeqLength;
  const int n = kHiddenDim * kHiddenSize / 2;
  half2 *out_ptr = reinterpret_cast<half2 *>(out);
  const half2 *bias_ptr = reinterpret_cast<const half2 *>(bias);

  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n;
       id += blockDim.x * gridDim.x) {
    half2 reg_bias = __ldg(&bias_ptr[id % n]);
    half2 val = out_ptr[id] + reg_bias;
    out_ptr[id] = gelu(val);
  }
}

__global__ void softmax(half *qk_buf_, const half *attr_mask,
                        const half scalar) {
  const int seq_id = blockIdx.x % kSeqLength;
  const int head_id = blockIdx.x / kSeqLength;
  const int warp_cols_num =
      kSeqLength / kWarpSize / (sizeof(half2) / sizeof(half));
  const int qk_offset =
      (((head_id * kSeqLength + seq_id) * kSeqLength) >> 1) + threadIdx.x;
  const int mask_offset = ((seq_id * kSeqLength) >> 1) + threadIdx.x;
  half2 *qk_buf_half2Ptr = reinterpret_cast<half2 *>(qk_buf_);
  const half2 *attr_mask_half2Ptr = reinterpret_cast<const half2 *>(attr_mask);

  half2 qk[warp_cols_num];
  float max_val = -1e20f;
  float sum_val = 0.0f;
  float mean_val;

#pragma unroll
  for (int i = 0; i < warp_cols_num; ++i) {
    half2 mask_val = __ldg(&attr_mask_half2Ptr[mask_offset + i * kWarpSize]);
    half2 mask_val_tmp = __hmul2(__hsub2(__float2half2_rn(1.0f), mask_val),
                                 __float2half2_rn(-10000.0f));
    qk[i] = qk_buf_half2Ptr[qk_offset + i * kWarpSize];
    qk[i] = __hadd2(__hmul2(__half2half2(scalar), qk[i]), mask_val_tmp);
    max_val = fmax(max_val, fmax((float)qk[i].x, (float)qk[i].y));
  }
  max_val = warpReduceMax(max_val);

#pragma unroll
  for (int i = 0; i < warp_cols_num; ++i) {
    qk[i] = h2exp(__hsub2(qk[i], __float2half2_rn(max_val)));
    sum_val += (float)(qk[i].x + qk[i].y);
  }
  sum_val = warpReduceSum(sum_val);
  mean_val = __fdividef(1.0f, sum_val + 1e-6f);

#pragma unroll
  for (int i = 0; i < warp_cols_num; ++i) {
    qk[i] = __hmul2(qk[i], __float2half2_rn(mean_val));
    qk_buf_half2Ptr[qk_offset + i * kWarpSize] = qk[i];
  }
}

__global__ void gemm_add_qkv_bias(const half *__restrict__ matrix_a,
                                  const half *__restrict__ matrix_b,
                                  const half *__restrict__ bias,
                                  half *__restrict__ matrix_c) {
  using namespace nvcuda;
  enum {
    kBlockRowTiles = kBlockRowWarps * kGemmK1WarpRowTiles,
    kBlockColTiles = kBlockColWarps * kGemmK1WarpColTiles,
  };

  extern __shared__ half all_shared_mem[];

  __attribute__((address_space(3))) half *matrix_a_shared[3][kStage],
      *matrix_b_shared[kStage];
  __attribute__((address_space(3))) half *acc_shared;

  matrix_a_shared[0][0] =
      (__attribute__((address_space(3))) half *)all_shared_mem;
  matrix_a_shared[0][1] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_a_shared[0][2] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_a_shared[1][0] =
      matrix_a_shared[0][0] +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_a_shared[1][1] =
      matrix_a_shared[0][1] +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_a_shared[1][2] =
      matrix_a_shared[0][2] +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_a_shared[2][0] =
      matrix_a_shared[1][0] +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_a_shared[2][1] =
      matrix_a_shared[1][1] +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_a_shared[2][2] =
      matrix_a_shared[1][2] +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

  matrix_b_shared[0] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      9 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_b_shared[1] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      9 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
      kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);
  matrix_b_shared[2] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      9 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
      2 * kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);

  acc_shared = (__attribute__((address_space(3))) half *)all_shared_mem;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_a[3][kGemmK1WarpRowTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_b[kGemmK1WarpColTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                         half>
      wmma_accumulator[3][kGemmK1WarpColTiles * kGemmK1WarpRowTiles];

  const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
  const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
  const int row_block_id = blockIdx.x % (kHiddenDim / kBlockRowTiles / kWmmaM);
  const int col_block_id = blockIdx.x / (kHiddenDim / kBlockRowTiles / kWmmaM);

#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int col = 0; col < kGemmK1WarpColTiles; ++col) {
#pragma unroll
      for (int row = 0; row < kGemmK1WarpRowTiles; ++row) {
        nvcuda::wmma::fill_fragment(
            wmma_accumulator[i][col * kGemmK1WarpRowTiles + row], 0.0f);
      }
    }
  }

  enum {
    kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
    kLoadALanesPerRow =
        kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
    kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

    kLoadBLanesPerRow = kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
    kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

    kAddBiasLanesPerRow =
        kWmmaM * kBlockRowTiles / (sizeof(half2) / sizeof(half)),
    kAddBiasColsPerIter = kThreads / kAddBiasLanesPerRow,

    kStoreCLanesPerRow = kLoadALanesPerRow,
    kStoreCColsPerIter = kLoadAColsPerIter,
  };

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  int stage = 0;
  int k_loop = 0;

  const int a_dst_stride =
      kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
  const int a_src_stride = kLoadAColsPerIter * kHiddenDim;

  const int b_dst_stride = kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
  const int b_src_stride = kLoadBColsPerIter * kHiddenDim;

#pragma unroll
  for (int s = 0; s < kStage - 1; ++s) {
    pipe.producer_acquire();
    __attribute__((address_space(3))) half *a_dst_base_0 =
        matrix_a_shared[0][(stage + s) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kBlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);
    __attribute__((address_space(3))) half *a_dst_base_1 =
        a_dst_base_0 +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    __attribute__((address_space(3))) half *a_dst_base_2 =
        a_dst_base_0 +
        6 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

    const half *a_src_base_0 =
        matrix_a + row_block_id * kBlockRowTiles * kWmmaM +
        ((k_loop + s) * kChunkK * kWmmaK + threadIdx.x / kLoadALanesPerRow) *
            kHiddenDim +
        (threadIdx.x & (kLoadALanesPerRow - 1)) *
            (sizeof(float4) / sizeof(half));
    const half *a_src_base_1 = a_src_base_0 + kHiddenDim * kHiddenDim;
    const half *a_src_base_2 = a_src_base_1 + kHiddenDim * kHiddenDim;

    __attribute__((address_space(3))) half *b_dst_base =
        matrix_b_shared[(stage + s) % kStage] +
        threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base = matrix_b + (k_loop + s) * kChunkK * kWmmaK +
                             (col_block_id * kBlockColTiles * kWmmaN +
                              threadIdx.x / kLoadBLanesPerRow) *
                                 kHiddenDim +
                             (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async((half *)a_dst_base_0 + i * a_dst_stride,
                         a_src_base_0 + i * a_src_stride, shape, pipe);
    }
#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async((half *)a_dst_base_1 + i * a_dst_stride,
                         a_src_base_1 + i * a_src_stride, shape, pipe);
    }
#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async((half *)a_dst_base_2 + i * a_dst_stride,
                         a_src_base_2 + i * a_src_stride, shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
      cuda::memcpy_async((half *)b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }
    pipe.producer_commit();
  }

#pragma unroll 10
  for (; k_loop < (kHiddenDim / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
    pipe.producer_acquire();

    __attribute__((address_space(3))) half *a_dst_base_0 =
        matrix_a_shared[0][(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kBlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);
    __attribute__((address_space(3))) half *a_dst_base_1 =
        a_dst_base_0 +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    __attribute__((address_space(3))) half *a_dst_base_2 =
        a_dst_base_0 +
        6 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    const half *a_src_base_0 = matrix_a +
                               row_block_id * kBlockRowTiles * kWmmaM +
                               ((k_loop + kStage - 1) * kChunkK * kWmmaK +
                                threadIdx.x / kLoadALanesPerRow) *
                                   kHiddenDim +
                               (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                   (sizeof(float4) / sizeof(half));
    const half *a_src_base_1 = a_src_base_0 + kHiddenDim * kHiddenDim;
    const half *a_src_base_2 = a_src_base_1 + kHiddenDim * kHiddenDim;

    __attribute__((address_space(3))) half *b_dst_base =
        matrix_b_shared[(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base = matrix_b +
                             (k_loop + kStage - 1) * kChunkK * kWmmaK +
                             (col_block_id * kBlockColTiles * kWmmaN +
                              threadIdx.x / kLoadBLanesPerRow) *
                                 kHiddenDim +
                             (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async((half *)a_dst_base_0 + i * a_dst_stride,
                         a_src_base_0 + i * a_src_stride, shape, pipe);
    }
#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async((half *)a_dst_base_1 + i * a_dst_stride,
                         a_src_base_1 + i * a_src_stride, shape, pipe);
    }
#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async((half *)a_dst_base_2 + i * a_dst_stride,
                         a_src_base_2 + i * a_src_stride, shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
      cuda::memcpy_async((half *)b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }
    pipe.producer_commit();

    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
      for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[0][tile_m],
            (half *)(matrix_a_shared[0][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[1][tile_m],
            (half *)(matrix_a_shared[1][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[2][tile_m],
            (half *)(matrix_a_shared[2][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
      }
#pragma unroll
      for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_b[tile_n],
            (half *)(matrix_b_shared[stage] +
                     (col_warp_id * kGemmK1WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
            kChunkK * kWmmaK + kInputSkew);
      }
#pragma unroll
      for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
          for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
            nvcuda::wmma::mma_sync(
                wmma_accumulator[i][tile_m + tile_n * kGemmK1WarpRowTiles],
                wmma_matrix_a[i][tile_m], wmma_matrix_b[tile_n],
                wmma_accumulator[i][tile_m + tile_n * kGemmK1WarpRowTiles]);
          }
        }
      }
    }
    stage = (stage + 1) % kStage;
  }

#pragma unroll
  for (int s = kStage - 1; s >= 1; --s) {
    k_loop = (kHiddenDim / kChunkK / kWmmaK) - s;
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
      for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[0][tile_m],
            (half *)(matrix_a_shared[0][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[1][tile_m],
            (half *)(matrix_a_shared[1][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[2][tile_m],
            (half *)(matrix_a_shared[2][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
      }
#pragma unroll
      for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_b[tile_n],
            (half *)(matrix_b_shared[stage] +
                     (col_warp_id * kGemmK1WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
            kChunkK * kWmmaK + kInputSkew);
      }
#pragma unroll
      for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
          for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
            nvcuda::wmma::mma_sync(
                wmma_accumulator[i][tile_m + tile_n * kGemmK1WarpRowTiles],
                wmma_matrix_a[i][tile_m], wmma_matrix_b[tile_n],
                wmma_accumulator[i][tile_m + tile_n * kGemmK1WarpRowTiles]);
          }
        }
      }
    }
    stage = (stage + 1) % kStage;
  }

#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
#pragma unroll
      for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
        nvcuda::wmma::store_matrix_sync(
            (half *)acc_shared +
                i * kBlockColTiles * kWmmaN *
                    (kBlockRowTiles * kWmmaM + kAccSkew) +
                (col_warp_id * kGemmK1WarpColTiles + tile_n) * kWmmaK *
                    (kBlockRowTiles * kWmmaM + kAccSkew) +
                (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM,
            wmma_accumulator[i][tile_n * kGemmK1WarpRowTiles + tile_m],
            (kBlockRowTiles * kWmmaM + kAccSkew), nvcuda::wmma::mem_col_major);
      }
    }
  }

  __syncthreads();

  const int bias_stride =
      kAddBiasColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);
  half *bias_dst_base =
      (half *)acc_shared +
      threadIdx.x / kAddBiasLanesPerRow * (kBlockRowTiles * kWmmaM + kAccSkew) +
      (threadIdx.x & (kAddBiasLanesPerRow - 1)) * sizeof(half2) / sizeof(half);
  const half *bias_src_base =
      bias + row_block_id * kBlockRowTiles * kWmmaM +
      (threadIdx.x & (kAddBiasLanesPerRow - 1)) * sizeof(half2) / sizeof(half);
#pragma unroll
  for (int j = 0; j < 3; ++j) {
#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kAddBiasColsPerIter; ++i) {
      *reinterpret_cast<half2 *>(bias_dst_base +
                                 j * kBlockColTiles * kWmmaN *
                                     (kBlockRowTiles * kWmmaM + kAccSkew) +
                                 i * bias_stride) +=
          __ldg(
              reinterpret_cast<const half2 *>(bias_src_base + j * kHiddenDim));
    }
  }

  __syncthreads();

  const int c_dst_stride = kStoreCColsPerIter * kHeadSize;
  const int c_src_stride =
      kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);

  half *c_dst_base =
      matrix_c + (row_block_id / 2) * 2 * kBlockRowTiles * kWmmaM * kSeqLength +
      (row_block_id % 2) * kBlockRowTiles * kWmmaM +
      (col_block_id * kBlockColTiles * kWmmaN +
       threadIdx.x / kStoreCLanesPerRow) *
          kHeadSize +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);
  half *c_src_base =
      (half *)acc_shared +
      threadIdx.x / kStoreCLanesPerRow * (kBlockRowTiles * kWmmaM + kAccSkew) +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

#pragma unroll
  for (int j = 0; j < 3; ++j) {
#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
      *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride +
                                  j * kHiddenDim * kSeqLength) =
          *reinterpret_cast<float4 *>(c_src_base + i * c_src_stride +
                                      j * kBlockColTiles * kWmmaN *
                                          (kBlockRowTiles * kWmmaM + kAccSkew));
    }
  }
}

__global__ void gemm_k2(const half *__restrict__ matrix_a,
                        const half *__restrict__ matrix_b,
                        half *__restrict__ matrix_c) {
  using namespace nvcuda;
  enum {
    kBlockRowTiles = kBlockRowWarps * kGemmK2WarpRowTiles,
    kBlockColTiles = kBlockColWarps * kGemmK2WarpColTiles,
  };

  extern __shared__ half all_shared_mem[];

  __attribute__((address_space(3))) half *matrix_a_shared =
      (__attribute__((address_space(3))) half *)all_shared_mem;

  __attribute__((address_space(3))) half *matrix_b_shared =
      matrix_a_shared + kBlockRowTiles * kWmmaM * (kHeadSize + kInputSkew);

  __attribute__((address_space(3))) half *acc_shared =
      (__attribute__((address_space(3))) half *)all_shared_mem;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::row_major>
      wmma_matrix_a[kGemmK2WarpRowTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_b[kGemmK2WarpColTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                         half>
      wmma_accumulator[kGemmK2WarpColTiles * kGemmK2WarpRowTiles];

  const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
  const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
  const int batch_stride = (kSeqLength / kBlockColTiles / kWmmaN) *
                           (kSeqLength / kBlockRowTiles / kWmmaM);
  const int batched_id = blockIdx.x / batch_stride;
  const int row_block_id =
      blockIdx.x % batch_stride % (kSeqLength / kBlockRowTiles / kWmmaM);
  const int col_block_id =
      blockIdx.x % batch_stride / (kSeqLength / kBlockRowTiles / kWmmaM);

#pragma unroll
  for (int col = 0; col < kGemmK2WarpColTiles; ++col) {
#pragma unroll
    for (int row = 0; row < kGemmK2WarpRowTiles; ++row) {
      nvcuda::wmma::fill_fragment(
          wmma_accumulator[col * kGemmK2WarpRowTiles + row], 0.0f);
    }
  }

  enum {
    kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
    kLoadLanesPerRow = kHeadSize / (sizeof(float4) / sizeof(half)),
    kLoadColsPerIter = kThreads / kLoadLanesPerRow,

    kStoreLanesPerRow =
        kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
    kStoreColsPerIter = kThreads / kStoreLanesPerRow,
  };

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));

  pipe.producer_acquire();
#pragma unroll
  for (int i = 0; i < kBlockRowTiles * kWmmaM / kLoadColsPerIter; ++i) {
    cuda::memcpy_async(
        reinterpret_cast<float4 *>(
            (half *)matrix_a_shared +
            (i * kLoadColsPerIter + threadIdx.x / kLoadLanesPerRow) *
                (kHeadSize + kInputSkew) +
            (threadIdx.x & (kLoadLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half)),
        reinterpret_cast<const float4 *>(
            matrix_a + batched_id * kSeqLength * kHeadSize +
            (row_block_id * kBlockRowTiles * kWmmaM + i * kLoadColsPerIter +
             threadIdx.x / kLoadLanesPerRow) *
                kHeadSize +
            (threadIdx.x & (kLoadLanesPerRow - 1)) *
                (sizeof(float4) / sizeof(half))),
        shape, pipe);
  }

#pragma unroll
  for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadColsPerIter; ++i) {
    cuda::memcpy_async(
        reinterpret_cast<float4 *>(
            (half *)matrix_b_shared +
            (i * kLoadColsPerIter + threadIdx.x / kLoadLanesPerRow) *
                (kHeadSize + kInputSkew) +
            (threadIdx.x & (kLoadLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half)),
        reinterpret_cast<const float4 *>(
            matrix_b + batched_id * kSeqLength * kHeadSize +
            (col_block_id * kBlockColTiles * kWmmaN + i * kLoadColsPerIter +
             threadIdx.x / kLoadLanesPerRow) *
                kHeadSize +
            (threadIdx.x & (kLoadLanesPerRow - 1)) *
                (sizeof(float4) / sizeof(half))),
        shape, pipe);
  }
  pipe.producer_commit();
  pipe.consumer_wait();
  __syncthreads();

#pragma unroll
  for (int tile_k = 0; tile_k < kHeadSize / kWmmaK; ++tile_k) {
#pragma unroll
    for (int tile_m = 0; tile_m < kGemmK2WarpRowTiles; ++tile_m) {
      nvcuda::wmma::load_matrix_sync(
          wmma_matrix_a[tile_m],
          (half *)(matrix_a_shared +
                   (row_warp_id * kGemmK2WarpRowTiles + tile_m) * kWmmaM *
                       (kHeadSize + kInputSkew) +
                   tile_k * kWmmaK),
          kHeadSize + kInputSkew);
    }
#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK2WarpColTiles; ++tile_n) {
      nvcuda::wmma::load_matrix_sync(
          wmma_matrix_b[tile_n],
          (half *)(matrix_b_shared +
                   (col_warp_id * kGemmK2WarpColTiles + tile_n) * kWmmaN *
                       (kHeadSize + kInputSkew) +
                   tile_k * kWmmaK),
          kHeadSize + kInputSkew);
    }
#pragma unroll
    for (int tile_m = 0; tile_m < kGemmK2WarpRowTiles; ++tile_m) {
#pragma unroll
      for (int tile_n = 0; tile_n < kGemmK2WarpColTiles; ++tile_n) {
        nvcuda::wmma::mma_sync(
            wmma_accumulator[tile_m + tile_n * kGemmK2WarpRowTiles],
            wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
            wmma_accumulator[tile_m + tile_n * kGemmK2WarpRowTiles]);
      }
    }
  }
  pipe.consumer_release();
  __syncthreads();

#pragma unroll
  for (int tile_n = 0; tile_n < kGemmK2WarpColTiles; ++tile_n) {
#pragma unroll
    for (int tile_m = 0; tile_m < kGemmK2WarpRowTiles; ++tile_m) {
      nvcuda::wmma::store_matrix_sync(
          (half *)acc_shared +
              (col_warp_id * kGemmK2WarpColTiles + tile_n) * kWmmaK *
                  (kBlockRowTiles * kWmmaM + kAccSkew) +
              (row_warp_id * kGemmK2WarpRowTiles + tile_m) * kWmmaM,
          wmma_accumulator[tile_n * kGemmK2WarpRowTiles + tile_m],
          (kBlockRowTiles * kWmmaM + kAccSkew), nvcuda::wmma::mem_col_major);
    }
  }

  __syncthreads();
#pragma unroll
  for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreColsPerIter; ++i) {
    *reinterpret_cast<float4 *>(
        matrix_c + batched_id * kSeqLength * kSeqLength +
        row_block_id * kBlockRowTiles * kWmmaM +
        (col_block_id * kBlockColTiles * kWmmaN + i * kStoreColsPerIter +
         threadIdx.x / kStoreLanesPerRow) *
            kSeqLength +
        (threadIdx.x & (kStoreLanesPerRow - 1)) * sizeof(float4) /
            sizeof(half)) =
        *reinterpret_cast<float4 *>(
            (half *)acc_shared +
            (i * kStoreColsPerIter + threadIdx.x / kStoreLanesPerRow) *
                (kBlockRowTiles * kWmmaM + kAccSkew) +
            (threadIdx.x & (kStoreLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half));
  }
}

__global__ void gemm_reshape(const half *__restrict__ matrix_a,
                             const half *__restrict__ matrix_b,
                             half *__restrict__ matrix_c) {
  using namespace nvcuda;
  enum {
    kBlockRowTiles = kBlockRowWarps * kGemmK3WarpRowTiles,
    kBlockColTiles = kBlockColWarps * kGemmK3WarpColTiles,
  };

  extern __shared__ half all_shared_mem[];

  __attribute__((address_space(3))) half *matrix_a_shared[kStage],
      *matrix_b_shared[kStage];
  __attribute__((address_space(3))) half *acc_shared;

  matrix_a_shared[0] = (__attribute__((address_space(3))) half *)all_shared_mem;
  matrix_a_shared[1] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_a_shared[2] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

  matrix_b_shared[0] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_b_shared[1] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
      kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);
  matrix_b_shared[2] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
      2 * kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);

  acc_shared = (__attribute__((address_space(3))) half *)all_shared_mem;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_a[kGemmK3WarpRowTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_b[kGemmK3WarpColTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                         half>
      wmma_accumulator[kGemmK3WarpColTiles * kGemmK3WarpRowTiles];

  const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
  const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
  const int batch_stride = kSeqLength / kBlockColTiles / kWmmaN;
  const int batched_id = blockIdx.x / batch_stride;
  const int col_block_id = blockIdx.x % batch_stride;

#pragma unroll
  for (int col = 0; col < kGemmK3WarpColTiles; ++col) {
#pragma unroll
    for (int row = 0; row < kGemmK3WarpRowTiles; ++row) {
      nvcuda::wmma::fill_fragment(
          wmma_accumulator[col * kGemmK3WarpRowTiles + row], 0.0f);
    }
  }

  enum {
    kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
    kLoadALanesPerRow =
        kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
    kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

    kLoadBLanesPerRow = kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
    kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

    kStoreCLanesPerRow = kLoadALanesPerRow,
    kStoreCColsPerIter = kLoadAColsPerIter,
  };

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  int stage = 0;
  int k_loop = 0;

  const int a_dst_stride =
      kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
  const int a_src_stride = kLoadAColsPerIter * kHeadSize;

  const int b_dst_stride = kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
  const int b_src_stride = kLoadBColsPerIter * kSeqLength;

  // Prologue
#pragma unroll
  for (int s = 0; s < kStage - 1; ++s) {
    pipe.producer_acquire();
    __attribute__((address_space(3))) half *a_dst_base =
        matrix_a_shared[(stage + s) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kBlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *a_src_base =
        matrix_a + batched_id * kSeqLength * kHeadSize +
        ((k_loop + s) * kChunkK * kWmmaK + threadIdx.x / kLoadALanesPerRow) *
            kHeadSize +
        (threadIdx.x & (kLoadALanesPerRow - 1)) *
            (sizeof(float4) / sizeof(half));

    __attribute__((address_space(3))) half *b_dst_base =
        matrix_b_shared[(stage + s) % kStage] +
        threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base = matrix_b + batched_id * kSeqLength * kSeqLength +
                             (k_loop + s) * kChunkK * kWmmaK +
                             (col_block_id * kBlockColTiles * kWmmaN +
                              threadIdx.x / kLoadBLanesPerRow) *
                                 kSeqLength +
                             (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async((half *)a_dst_base + i * a_dst_stride,
                         a_src_base + i * a_src_stride, shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
      cuda::memcpy_async((half *)b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }
    pipe.producer_commit();
  }

  // Soft pipeline
#pragma unroll 4
  for (; k_loop < (kSeqLength / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
    pipe.producer_acquire();

    __attribute__((address_space(3))) half *a_dst_base =
        matrix_a_shared[(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kBlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *a_src_base = matrix_a + batched_id * kSeqLength * kHeadSize +
                             ((k_loop + kStage - 1) * kChunkK * kWmmaK +
                              threadIdx.x / kLoadALanesPerRow) *
                                 kHeadSize +
                             (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

    __attribute__((address_space(3))) half *b_dst_base =
        matrix_b_shared[(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base = matrix_b + batched_id * kSeqLength * kSeqLength +
                             (k_loop + kStage - 1) * kChunkK * kWmmaK +
                             (col_block_id * kBlockColTiles * kWmmaN +
                              threadIdx.x / kLoadBLanesPerRow) *
                                 kSeqLength +
                             (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async((half *)a_dst_base + i * a_dst_stride,
                         a_src_base + i * a_src_stride, shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
      cuda::memcpy_async((half *)b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }
    pipe.producer_commit();

    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
      for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[tile_m],
            (half *)(matrix_a_shared[stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK3WarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
      }
#pragma unroll
      for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_b[tile_n],
            (half *)(matrix_b_shared[stage] +
                     (col_warp_id * kGemmK3WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
            kChunkK * kWmmaK + kInputSkew);
      }
#pragma unroll
      for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
#pragma unroll
        for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
          nvcuda::wmma::mma_sync(
              wmma_accumulator[tile_m + tile_n * kGemmK3WarpRowTiles],
              wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
              wmma_accumulator[tile_m + tile_n * kGemmK3WarpRowTiles]);
        }
      }
    }
    stage = (stage + 1) % kStage;
  }

  // Epilogue
#pragma unroll
  for (int s = kStage - 1; s >= 1; --s) {
    k_loop = (kSeqLength / kChunkK / kWmmaK) - s;
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
      for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[tile_m],
            (half *)(matrix_a_shared[stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK3WarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
      }
#pragma unroll
      for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_b[tile_n],
            (half *)(matrix_b_shared[stage] +
                     (col_warp_id * kGemmK3WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
            kChunkK * kWmmaK + kInputSkew);
      }
#pragma unroll
      for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
#pragma unroll
        for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
          nvcuda::wmma::mma_sync(
              wmma_accumulator[tile_m + tile_n * kGemmK3WarpRowTiles],
              wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
              wmma_accumulator[tile_m + tile_n * kGemmK3WarpRowTiles]);
        }
      }
    }
    stage = (stage + 1) % kStage;
  }

#pragma unroll
  for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
#pragma unroll
    for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
      nvcuda::wmma::store_matrix_sync(
          (half *)acc_shared +
              (col_warp_id * kGemmK3WarpColTiles + tile_n) * kWmmaK *
                  (kBlockRowTiles * kWmmaM + kAccSkew) +
              (row_warp_id * kGemmK3WarpRowTiles + tile_m) * kWmmaM,
          wmma_accumulator[tile_n * kGemmK3WarpRowTiles + tile_m],
          (kBlockRowTiles * kWmmaM + kAccSkew), nvcuda::wmma::mem_col_major);
    }
  }

  __syncthreads();

  const int c_dst_stride = kStoreCColsPerIter * kHiddenDim;
  const int c_src_stride =
      kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);

  half *c_dst_base =
      matrix_c + batched_id * kHeadSize +
      (col_block_id * kBlockColTiles * kWmmaN +
       threadIdx.x / kStoreCLanesPerRow) *
          kHiddenDim +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);
  half *c_src_base =
      (half *)acc_shared +
      threadIdx.x / kStoreCLanesPerRow * (kBlockRowTiles * kWmmaM + kAccSkew) +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

#pragma unroll
  for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
    *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride) =
        *reinterpret_cast<float4 *>(c_src_base + i * c_src_stride);
  }
}

template <int kWarpRowTiles, int kWarpColTiles, int M, int N, int K, int B>
__global__ void gemm_three_stage(const half *__restrict__ matrix_a,
                                 const half *__restrict__ matrix_b,
                                 half *__restrict__ matrix_c) {
  using namespace nvcuda;
  enum {
    kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
    kBlockColTiles = kBlockColWarps * kWarpColTiles,
  };

  extern __shared__ half all_shared_mem[];

  __attribute__((address_space(3))) half *matrix_a_shared[kStage],
      *matrix_b_shared[kStage];
  __attribute__((address_space(3))) half *acc_shared;

  matrix_a_shared[0] = (__attribute__((address_space(3))) half *)all_shared_mem;
  matrix_a_shared[1] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_a_shared[2] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

  matrix_b_shared[0] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_b_shared[1] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
      kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);
  matrix_b_shared[2] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
      2 * kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);

  acc_shared = (__attribute__((address_space(3))) half *)all_shared_mem;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_a[kWarpRowTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_b[kWarpColTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                         half>
      wmma_accumulator[kWarpColTiles * kWarpRowTiles];

  const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
  const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
  const int batch_stride =
      (N / kBlockColTiles / kWmmaN) * (M / kBlockRowTiles / kWmmaM);
  const int batched_id = blockIdx.x / batch_stride;
  const int row_block_id =
      blockIdx.x % batch_stride % (M / kBlockRowTiles / kWmmaM);
  const int col_block_id =
      blockIdx.x % batch_stride / (M / kBlockRowTiles / kWmmaM);

#pragma unroll
  for (int col = 0; col < kWarpColTiles; ++col) {
#pragma unroll
    for (int row = 0; row < kWarpRowTiles; ++row) {
      nvcuda::wmma::fill_fragment(wmma_accumulator[col * kWarpRowTiles + row],
                                  0.0f);
    }
  }

  enum {
    kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
    kLoadALanesPerRow =
        kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
    kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

    kLoadBLanesPerRow = kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
    kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

    kStoreCLanesPerRow = kLoadALanesPerRow,
    kStoreCColsPerIter = kLoadAColsPerIter,
  };

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  int stage = 0;
  int k_loop = 0;

  const int a_dst_stride =
      kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
  const int a_src_stride = kLoadAColsPerIter * M;

  const int b_dst_stride = kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
  const int b_src_stride = kLoadBColsPerIter * K;

  // Prologue
#pragma unroll
  for (int s = 0; s < kStage - 1; ++s) {
    pipe.producer_acquire();
    __attribute__((address_space(3))) half *a_dst_base =
        matrix_a_shared[(stage + s) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kBlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *a_src_base =
        matrix_a + batched_id * K * M + row_block_id * kBlockRowTiles * kWmmaM +
        ((k_loop + s) * kChunkK * kWmmaK + threadIdx.x / kLoadALanesPerRow) *
            M +
        (threadIdx.x & (kLoadALanesPerRow - 1)) *
            (sizeof(float4) / sizeof(half));

    __attribute__((address_space(3))) half *b_dst_base =
        matrix_b_shared[(stage + s) % kStage] +
        threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base = matrix_b + batched_id * N * K +
                             (k_loop + s) * kChunkK * kWmmaK +
                             (col_block_id * kBlockColTiles * kWmmaN +
                              threadIdx.x / kLoadBLanesPerRow) *
                                 K +
                             (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async((half *)a_dst_base + i * a_dst_stride,
                         a_src_base + i * a_src_stride, shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
      cuda::memcpy_async((half *)b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }
    pipe.producer_commit();
  }

  // Soft pipeline
#pragma unroll(K / 64 - 2)
  for (; k_loop < (K / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
    pipe.producer_acquire();

    __attribute__((address_space(3))) half *a_dst_base =
        matrix_a_shared[(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kBlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *a_src_base = matrix_a + batched_id * K * M +
                             row_block_id * kBlockRowTiles * kWmmaM +
                             ((k_loop + kStage - 1) * kChunkK * kWmmaK +
                              threadIdx.x / kLoadALanesPerRow) *
                                 M +
                             (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

    __attribute__((address_space(3))) half *b_dst_base =
        matrix_b_shared[(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base = matrix_b + batched_id * N * K +
                             (k_loop + kStage - 1) * kChunkK * kWmmaK +
                             (col_block_id * kBlockColTiles * kWmmaN +
                              threadIdx.x / kLoadBLanesPerRow) *
                                 K +
                             (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async((half *)a_dst_base + i * a_dst_stride,
                         a_src_base + i * a_src_stride, shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
      cuda::memcpy_async((half *)b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }
    pipe.producer_commit();

    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
      for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[tile_m],
            (half *)(matrix_a_shared[stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
      }
#pragma unroll
      for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_b[tile_n],
            (half *)(matrix_b_shared[stage] +
                     (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
            kChunkK * kWmmaK + kInputSkew);
      }
#pragma unroll
      for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
          nvcuda::wmma::mma_sync(
              wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
              wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
              wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
        }
      }
    }
    stage = (stage + 1) % kStage;
  }

  // Epilogue
#pragma unroll
  for (int s = kStage - 1; s >= 1; --s) {
    k_loop = (K / kChunkK / kWmmaK) - s;
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
      for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[tile_m],
            (half *)(matrix_a_shared[stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
      }
#pragma unroll
      for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_b[tile_n],
            (half *)(matrix_b_shared[stage] +
                     (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
            kChunkK * kWmmaK + kInputSkew);
      }
#pragma unroll
      for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
          nvcuda::wmma::mma_sync(
              wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
              wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
              wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
        }
      }
    }
    stage = (stage + 1) % kStage;
  }

#pragma unroll
  for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
#pragma unroll
    for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
      nvcuda::wmma::store_matrix_sync(
          (half *)acc_shared +
              (col_warp_id * kWarpColTiles + tile_n) * kWmmaK *
                  (kBlockRowTiles * kWmmaM + kAccSkew) +
              (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM,
          wmma_accumulator[tile_n * kWarpRowTiles + tile_m],
          (kBlockRowTiles * kWmmaM + kAccSkew), nvcuda::wmma::mem_col_major);
    }
  }

  __syncthreads();

  const int c_dst_stride = kStoreCColsPerIter * M;
  const int c_src_stride =
      kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);

  half *c_dst_base =
      matrix_c + batched_id * N * M + row_block_id * kBlockRowTiles * kWmmaM +
      (col_block_id * kBlockColTiles * kWmmaN +
       threadIdx.x / kStoreCLanesPerRow) *
          M +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);
  half *c_src_base =
      (half *)acc_shared +
      threadIdx.x / kStoreCLanesPerRow * (kBlockRowTiles * kWmmaM + kAccSkew) +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

#pragma unroll
  for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
    *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride) =
        *reinterpret_cast<float4 *>(c_src_base + i * c_src_stride);
  }
}

__global__ void gemm_k6(const half *__restrict__ matrix_a,
                        const half *__restrict__ matrix_b,
                        half *__restrict__ matrix_c) {
  using namespace nvcuda;

  extern __shared__ half all_shared_mem[];
  //__shared__ half all_shared_mem[24576];

  __attribute__((address_space(3))) half *matrix_a_shared[kStage],
      *matrix_b_shared[kStage];
  __attribute__((address_space(3))) half *acc_shared;

  matrix_a_shared[0] = (__attribute__((address_space(3))) half *)all_shared_mem;
  matrix_a_shared[1] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      kGemmK6BlockSliceKTiles * kWmmaK *
          (kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
  matrix_a_shared[2] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      2 * kGemmK6BlockSliceKTiles * kWmmaK *
          (kGemmK6BlockRowTiles * kWmmaM + kInputSkew);

  matrix_b_shared[0] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      3 * kGemmK6BlockSliceKTiles * kWmmaK *
          (kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
  matrix_b_shared[1] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      3 * kGemmK6BlockSliceKTiles * kWmmaK *
          (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
      kGemmK6BlockColTiles * kWmmaN *
          (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew);
  matrix_b_shared[2] =
      (__attribute__((address_space(3))) half *)all_shared_mem +
      3 * kGemmK6BlockSliceKTiles * kWmmaK *
          (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
      2 * kGemmK6BlockColTiles * kWmmaN *
          (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew);

  acc_shared = (__attribute__((address_space(3))) half *)all_shared_mem;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_a[kGemmK6BlockRowTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_b[kGemmK6BlockColTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                         half>
      wmma_accumulator[kGemmK6BlockRowTiles * kGemmK6BlockColTiles];

  const int slicek_warp_id = threadIdx.x / kWarpSize;
  const int row_block_id =
      blockIdx.x % (kHiddenDim / kGemmK6BlockRowTiles / kWmmaM);
  const int col_block_id =
      blockIdx.x / (kHiddenDim / kGemmK6BlockRowTiles / kWmmaM);

#pragma unroll
  for (int col = 0; col < kGemmK6BlockColTiles; ++col) {
#pragma unroll
    for (int row = 0; row < kGemmK6BlockRowTiles; ++row) {
      nvcuda::wmma::fill_fragment(
          wmma_accumulator[col * kGemmK6BlockRowTiles + row], 0.0f);
    }
  }

  enum {
    kThreads = kGemmK6BlockSliceKTiles * kWarpSize,
    kLoadALanesPerRow =
        kWmmaM * kGemmK6BlockRowTiles / (sizeof(float4) / sizeof(half)),
    kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

    kLoadBLanesPerRow =
        kWmmaK * kGemmK6BlockSliceKTiles / (sizeof(float4) / sizeof(half)),
    kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

    kReduceCLanesPerRow =
        kWmmaM * kGemmK6BlockRowTiles / (sizeof(half2) / sizeof(half)),
    kReduceCColsPerIter = kThreads / kReduceCLanesPerRow,

    kStoreCLanesPerRow = kLoadALanesPerRow,
    kStoreCColsPerIter = kLoadAColsPerIter,
  };

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  int stage = 0;
  int k_loop = 0;

  const int a_dst_stride =
      kLoadAColsPerIter * (kWmmaM * kGemmK6BlockRowTiles + kInputSkew);
  const int a_src_stride = kLoadAColsPerIter * kHiddenDim;

  const int b_dst_stride =
      kLoadBColsPerIter * (kWmmaK * kGemmK6BlockSliceKTiles + kInputSkew);
  const int b_src_stride = kLoadBColsPerIter * kHiddenDim * kHiddenSize;

  // Prologue
#pragma unroll
  for (int s = 0; s < kStage - 1; ++s) {
    pipe.producer_acquire();
    __attribute__((address_space(3))) half *a_dst_base =
        matrix_a_shared[(stage + s) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kGemmK6BlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *a_src_base = matrix_a +
                             row_block_id * kGemmK6BlockRowTiles * kWmmaM +
                             ((k_loop + s) * kGemmK6BlockSliceKTiles * kWmmaK +
                              threadIdx.x / kLoadALanesPerRow) *
                                 kHiddenDim +
                             (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

    __attribute__((address_space(3))) half *b_dst_base =
        matrix_b_shared[(stage + s) % kStage] +
        threadIdx.x / kLoadBLanesPerRow *
            (kWmmaK * kGemmK6BlockSliceKTiles + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base = matrix_b +
                             (k_loop + s) * kGemmK6BlockSliceKTiles * kWmmaK +
                             (col_block_id * kGemmK6BlockColTiles * kWmmaN +
                              threadIdx.x / kLoadBLanesPerRow) *
                                 kHiddenDim * kHiddenSize +
                             (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kGemmK6BlockSliceKTiles * kWmmaK / kLoadAColsPerIter;
         ++i) {
      cuda::memcpy_async((half *)a_dst_base + i * a_dst_stride,
                         a_src_base + i * a_src_stride, shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kLoadBColsPerIter;
         ++i) {
      cuda::memcpy_async((half *)b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }
    pipe.producer_commit();
  }

  // Soft pipeline
#pragma unroll 46
  for (;
       k_loop < (kHiddenDim * kHiddenSize / kGemmK6BlockSliceKTiles / kWmmaK) -
                    (kStage - 1);
       ++k_loop) {
    pipe.producer_acquire();

    __attribute__((address_space(3))) half *a_dst_base =
        matrix_a_shared[(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kGemmK6BlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *a_src_base =
        matrix_a + row_block_id * kGemmK6BlockRowTiles * kWmmaM +
        ((k_loop + kStage - 1) * kGemmK6BlockSliceKTiles * kWmmaK +
         threadIdx.x / kLoadALanesPerRow) *
            kHiddenDim +
        (threadIdx.x & (kLoadALanesPerRow - 1)) *
            (sizeof(float4) / sizeof(half));

    __attribute__((address_space(3))) half *b_dst_base =
        matrix_b_shared[(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadBLanesPerRow *
            (kWmmaK * kGemmK6BlockSliceKTiles + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base =
        matrix_b + (k_loop + kStage - 1) * kGemmK6BlockSliceKTiles * kWmmaK +
        (col_block_id * kGemmK6BlockColTiles * kWmmaN +
         threadIdx.x / kLoadBLanesPerRow) *
            kHiddenDim * kHiddenSize +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) *
            (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kGemmK6BlockSliceKTiles * kWmmaK / kLoadAColsPerIter;
         ++i) {
      cuda::memcpy_async((half *)a_dst_base + i * a_dst_stride,
                         a_src_base + i * a_src_stride, shape, pipe);
    }
#pragma unroll
    for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kLoadBColsPerIter;
         ++i) {
      cuda::memcpy_async((half *)b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }

    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
      nvcuda::wmma::load_matrix_sync(
          wmma_matrix_a[tile_m],
          (half *)(matrix_a_shared[stage] +
                   slicek_warp_id * kWmmaK *
                       (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
                   tile_m * kWmmaM),
          kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
    }
#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
      nvcuda::wmma::load_matrix_sync(
          wmma_matrix_b[tile_n],
          (half *)(matrix_b_shared[stage] +
                   tile_n * kWmmaN *
                       (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew) +
                   slicek_warp_id * kWmmaK),
          kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew);
    }
#pragma unroll
    for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
#pragma unroll
      for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
        nvcuda::wmma::mma_sync(
            wmma_accumulator[tile_m + tile_n * kGemmK6BlockRowTiles],
            wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
            wmma_accumulator[tile_m + tile_n * kGemmK6BlockRowTiles]);
      }
    }
    stage = (stage + 1) % kStage;
  }

#pragma unroll
  for (int s = kStage - 1; s >= 1; --s) {
    k_loop = (kHiddenDim * kHiddenSize / kGemmK6BlockSliceKTiles / kWmmaK) - s;
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
      nvcuda::wmma::load_matrix_sync(
          wmma_matrix_a[tile_m],
          (half *)(matrix_a_shared[stage] +
                   slicek_warp_id * kWmmaK *
                       (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
                   tile_m * kWmmaM),
          kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
    }
#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
      nvcuda::wmma::load_matrix_sync(
          wmma_matrix_b[tile_n],
          (half *)(matrix_b_shared[stage] +
                   tile_n * kWmmaN *
                       (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew) +
                   slicek_warp_id * kWmmaK),
          kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew);
    }
#pragma unroll
    for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
#pragma unroll
      for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
        nvcuda::wmma::mma_sync(
            wmma_accumulator[tile_m + tile_n * kGemmK6BlockRowTiles],
            wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
            wmma_accumulator[tile_m + tile_n * kGemmK6BlockRowTiles]);
      }
    }
    stage = (stage + 1) % kStage;
  }

#pragma unroll
  for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
      nvcuda::wmma::store_matrix_sync(
          (half *)(acc_shared +
                   (slicek_warp_id * kGemmK6BlockColTiles + tile_n) * kWmmaN *
                       (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) +
                   tile_m * kWmmaM),
          wmma_accumulator[tile_n * kGemmK6BlockRowTiles + tile_m],
          (kGemmK6BlockRowTiles * kWmmaM + kAccSkew),
          nvcuda::wmma::mem_col_major);
    }
  }

  __syncthreads();

  const int c_reduce_stride =
      kReduceCColsPerIter * (kGemmK6BlockRowTiles * kWmmaM + kAccSkew);
  const int c_reduce_k_stride = kGemmK6BlockColTiles * kWmmaN *
                                (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) *
                                sizeof(half) / sizeof(half2);
  half *c_reduce_base =
      (half *)acc_shared +
      threadIdx.x / kReduceCLanesPerRow *
          (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) +
      (threadIdx.x & (kReduceCLanesPerRow - 1)) * sizeof(half2) / sizeof(half);
#pragma unroll
  for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kReduceCColsPerIter;
       ++i) {
    half2 *c_reduce_src =
        reinterpret_cast<half2 *>(c_reduce_base + i * c_reduce_stride);
#pragma unroll
    for (int k = 1; k < kGemmK6BlockSliceKTiles; ++k) {
      *c_reduce_src += *(c_reduce_src + k * c_reduce_k_stride);
    }
  }
  __syncthreads();

  const int c_dst_stride = kStoreCColsPerIter * kHiddenDim;
  const int c_src_stride =
      kStoreCColsPerIter * (kGemmK6BlockRowTiles * kWmmaM + kAccSkew);

  half *c_dst_base =
      matrix_c + row_block_id * kGemmK6BlockRowTiles * kWmmaM +
      (col_block_id * kGemmK6BlockColTiles * kWmmaN +
       threadIdx.x / kStoreCLanesPerRow) *
          kHiddenDim +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);
  half *c_src_base =
      (half *)acc_shared +
      threadIdx.x / kStoreCLanesPerRow *
          (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

#pragma unroll
  for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
    *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride) =
        *reinterpret_cast<float4 *>(c_src_base + i * c_src_stride);
  }
}
