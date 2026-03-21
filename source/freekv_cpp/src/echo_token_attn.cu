#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <cfloat>

#include "pytorch_extension_utils.h"

namespace {

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v);

template <>
__device__ inline float scalar_to_float<nv_half>(nv_half v) {
  return __half2float(v);
}

#ifdef FLASHINFER_ENABLE_BF16
template <>
__device__ inline float scalar_to_float<nv_bfloat16>(nv_bfloat16 v) {
  return __bfloat162float(v);
}
#endif

template <int THREADS>
__device__ inline float block_reduce_sum(float v) {
  __shared__ float smem[THREADS];
  smem[threadIdx.x] = v;
  __syncthreads();
  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smem[threadIdx.x] += smem[threadIdx.x + s];
    }
    __syncthreads();
  }
  return smem[0];
}

template <int THREADS>
__device__ inline int block_reduce_sum_int(int v) {
  __shared__ int smem[THREADS];
  smem[threadIdx.x] = v;
  __syncthreads();
  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smem[threadIdx.x] += smem[threadIdx.x + s];
    }
    __syncthreads();
  }
  return smem[0];
}

template <int THREADS>
__device__ inline float block_reduce_max(float v) {
  __shared__ float smem[THREADS];
  smem[threadIdx.x] = v;
  __syncthreads();
  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    }
    __syncthreads();
  }
  return smem[0];
}

template <typename scalar_t, int THREADS>
__global__ void echo_qk_scores_chunk_kernel(
    const scalar_t* __restrict__ q_ptr,      // [b, hq, d]
    const scalar_t* __restrict__ k_ptr,      // [b, hk, s, d]
    float* __restrict__ scores_ptr,          // [b, hq, max_t]
    int64_t stride_q_b,
    int64_t stride_q_h,
    int64_t stride_q_d,
    int64_t stride_k_b,
    int64_t stride_k_h,
    int64_t stride_k_t,
    int64_t stride_k_d,
    int64_t stride_s_b,
    int64_t stride_s_h,
    int64_t stride_s_t,
    int64_t n_q_heads,
    int64_t n_q_per_kv,
    int64_t head_dim,
    int64_t token_begin,
    int64_t token_count,
    float scale) {
  int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t tok_local = linear % token_count;
  linear /= token_count;
  const int64_t qh = linear % n_q_heads;
  const int64_t b = linear / n_q_heads;
  const int64_t kvh = qh / n_q_per_kv;
  const int64_t tok = token_begin + tok_local;

  const scalar_t* q_base = q_ptr + b * stride_q_b + qh * stride_q_h;
  const scalar_t* k_base = k_ptr + b * stride_k_b + kvh * stride_k_h + tok * stride_k_t;

  float local_sum = 0.0f;
  for (int64_t d = threadIdx.x; d < head_dim; d += THREADS) {
    const float qv = scalar_to_float<scalar_t>(q_base[d * stride_q_d]);
    const float kv = scalar_to_float<scalar_t>(k_base[d * stride_k_d]);
    local_sum += qv * kv;
  }
  const float sum = block_reduce_sum<THREADS>(local_sum);
  if (threadIdx.x == 0) {
    const int64_t s_off = b * stride_s_b + qh * stride_s_h + tok * stride_s_t;
    scores_ptr[s_off] = sum * scale;
  }
}

template <typename scalar_t, int THREADS>
__global__ void echo_qk_pagemax_chunk_only_kernel(
    const scalar_t* __restrict__ q_ptr,      // [b, hq, d]
    const scalar_t* __restrict__ k_ptr,      // [b, hk, s, d]
    int32_t* __restrict__ out_best_ptr,      // [b, hq, page_count]
    int64_t stride_q_b,
    int64_t stride_q_h,
    int64_t stride_q_d,
    int64_t stride_k_b,
    int64_t stride_k_h,
    int64_t stride_k_t,
    int64_t stride_k_d,
    int64_t stride_o_b,
    int64_t stride_o_h,
    int64_t stride_o_p,
    int64_t n_q_heads,
    int64_t n_q_per_kv,
    int64_t head_dim,
    int64_t token_begin,
    int64_t page_size,
    int64_t page_count) {
  int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t p = linear % page_count;
  linear /= page_count;
  const int64_t qh = linear % n_q_heads;
  const int64_t b = linear / n_q_heads;
  const int64_t kvh = qh / n_q_per_kv;

  const int64_t tok_local = static_cast<int64_t>(threadIdx.x);
  float score = -FLT_MAX;
  int32_t idx = static_cast<int32_t>(tok_local);
  if (tok_local < page_size) {
    const int64_t tok = token_begin + p * page_size + tok_local;
    const scalar_t* q_base = q_ptr + b * stride_q_b + qh * stride_q_h;
    const scalar_t* k_base = k_ptr + b * stride_k_b + kvh * stride_k_h + tok * stride_k_t;
    float acc = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      const float qv = scalar_to_float<scalar_t>(q_base[d * stride_q_d]);
      const float kv = scalar_to_float<scalar_t>(k_base[d * stride_k_d]);
      acc += qv * kv;
    }
    score = acc;
  }

  __shared__ float s_best_v[THREADS];
  __shared__ int32_t s_best_i[THREADS];
  s_best_v[threadIdx.x] = score;
  s_best_i[threadIdx.x] = idx;
  __syncthreads();

  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = s_best_v[threadIdx.x + s];
      const int32_t oi = s_best_i[threadIdx.x + s];
      if ((ov > s_best_v[threadIdx.x]) ||
          ((ov == s_best_v[threadIdx.x]) && (oi < s_best_i[threadIdx.x]))) {
        s_best_v[threadIdx.x] = ov;
        s_best_i[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out_best_ptr[b * stride_o_b + qh * stride_o_h + p * stride_o_p] = s_best_i[0];
  }
}

template <int THREADS>
__global__ void echo_reduce_qh_best_mean_kernel(
    const int32_t* __restrict__ in_best_ptr,   // [b, hq, p]
    int32_t* __restrict__ out_best_ptr,        // [b, p]
    int64_t stride_i_b,
    int64_t stride_i_h,
    int64_t stride_i_p,
    int64_t stride_o_b,
    int64_t stride_o_p,
    int64_t n_q_heads,
    int64_t page_count) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t p = linear % page_count;
  const int64_t b = linear / page_count;

  int local_sum = 0;
  for (int64_t qh = threadIdx.x; qh < n_q_heads; qh += THREADS) {
    local_sum += static_cast<int>(
        in_best_ptr[b * stride_i_b + qh * stride_i_h + p * stride_i_p]);
  }
  const int sum = block_reduce_sum_int<THREADS>(local_sum);
  if (threadIdx.x == 0) {
    const int rounded_mean = (sum + static_cast<int>(n_q_heads / 2)) /
                             static_cast<int>(n_q_heads);
    out_best_ptr[b * stride_o_b + p * stride_o_p] = static_cast<int32_t>(rounded_mean);
  }
}

template <int THREADS>
__global__ void echo_page_argmax_from_scores_kernel(
    const float* __restrict__ scores_ptr,    // [b, hq, max_t]
    int32_t* __restrict__ out_best_ptr,      // [b, hq, page_count]
    int64_t stride_s_b,
    int64_t stride_s_h,
    int64_t stride_s_t,
    int64_t stride_o_b,
    int64_t stride_o_h,
    int64_t stride_o_p,
    int64_t n_q_heads,
    int64_t token_begin,
    int64_t page_size,
    int64_t page_count) {
  int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t p = linear % page_count;
  linear /= page_count;
  const int64_t qh = linear % n_q_heads;
  const int64_t b = linear / n_q_heads;

  const int64_t page_tok_base = token_begin + p * page_size;
  float best_v = -FLT_MAX;
  int32_t best_i = 0;
  for (int64_t t = threadIdx.x; t < page_size; t += THREADS) {
    const int64_t tok = page_tok_base + t;
    const float v = scores_ptr[b * stride_s_b + qh * stride_s_h + tok * stride_s_t];
    if (v > best_v) {
      best_v = v;
      best_i = static_cast<int32_t>(t);
    }
  }

  __shared__ float s_best_v[THREADS];
  __shared__ int32_t s_best_i[THREADS];
  s_best_v[threadIdx.x] = best_v;
  s_best_i[threadIdx.x] = best_i;
  __syncthreads();

  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = s_best_v[threadIdx.x + s];
      const int32_t oi = s_best_i[threadIdx.x + s];
      if ((ov > s_best_v[threadIdx.x]) ||
          ((ov == s_best_v[threadIdx.x]) && (oi < s_best_i[threadIdx.x]))) {
        s_best_v[threadIdx.x] = ov;
        s_best_i[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out_best_ptr[b * stride_o_b + qh * stride_o_h + p * stride_o_p] = s_best_i[0];
  }
}

template <typename scalar_t, int THREADS>
__global__ void echo_pv_from_scores_kernel(
    const float* __restrict__ scores_ptr,    // [b, hq, max_t]
    const scalar_t* __restrict__ v_ptr,      // [b, hk, seq, d]
    float* __restrict__ out_ptr,             // [b, hq, d], float32
    int64_t stride_s_b,
    int64_t stride_s_h,
    int64_t stride_s_t,
    int64_t stride_v_b,
    int64_t stride_v_h,
    int64_t stride_v_t,
    int64_t stride_v_d,
    int64_t stride_o_b,
    int64_t stride_o_h,
    int64_t stride_o_d,
    int64_t n_q_heads,
    int64_t n_q_per_kv,
    int64_t seq_len,
    int64_t head_dim) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t qh = linear % n_q_heads;
  const int64_t b = linear / n_q_heads;
  const int64_t kvh = qh / n_q_per_kv;

  const float* s_base = scores_ptr + b * stride_s_b + qh * stride_s_h;
  const scalar_t* v_base = v_ptr + b * stride_v_b + kvh * stride_v_h;

  float local_max = -FLT_MAX;
  for (int64_t t = threadIdx.x; t < seq_len; t += THREADS) {
    local_max = fmaxf(local_max, s_base[t * stride_s_t]);
  }
  const float max_v = block_reduce_max<THREADS>(local_max);

  float local_sum = 0.0f;
  for (int64_t t = threadIdx.x; t < seq_len; t += THREADS) {
    local_sum += expf(s_base[t * stride_s_t] - max_v);
  }
  float denom = block_reduce_sum<THREADS>(local_sum);
  if (denom <= 0.0f) {
    denom = 1.0f;
  }
  const float inv_denom = 1.0f / denom;

  for (int64_t d = threadIdx.x; d < head_dim; d += THREADS) {
    float acc = 0.0f;
    for (int64_t t = 0; t < seq_len; ++t) {
      const float p = expf(s_base[t * stride_s_t] - max_v) * inv_denom;
      const float vv = scalar_to_float<scalar_t>(v_base[t * stride_v_t + d * stride_v_d]);
      acc += p * vv;
    }
    out_ptr[b * stride_o_b + qh * stride_o_h + d * stride_o_d] = acc;
  }
}

template <typename scalar_t>
void launch_qk_scores_chunk(
    const torch::Tensor& q,
    const torch::Tensor& k,
    torch::Tensor scores,
    int64_t n_q_per_kv,
    int64_t token_begin,
    int64_t token_count) {
  if (token_count == 0) {
    return;
  }
  constexpr int THREADS = 128;
  const int64_t bsz = q.size(0);
  const int64_t n_q_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t total = bsz * n_q_heads * token_count;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  echo_qk_scores_chunk_kernel<scalar_t, THREADS>
      <<<static_cast<uint32_t>(total), THREADS, 0, stream>>>(
          reinterpret_cast<const scalar_t*>(q.data_ptr()),
          reinterpret_cast<const scalar_t*>(k.data_ptr()),
          scores.data_ptr<float>(),
          q.stride(0),
          q.stride(1),
          q.stride(2),
          k.stride(0),
          k.stride(1),
          k.stride(2),
          k.stride(3),
          scores.stride(0),
          scores.stride(1),
          scores.stride(2),
          n_q_heads,
          n_q_per_kv,
          head_dim,
          token_begin,
          token_count,
          scale);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "echo_qk_scores_chunk_kernel launch failed: ", cudaGetErrorString(err));
}

}  // namespace

void echo_decode_qk_scores_chunk(
    const torch::Tensor &q,
    const torch::Tensor &k,
    torch::Tensor scores,
    int64_t n_q_per_kv,
    int64_t token_begin,
    int64_t token_count) {
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(scores);
  TORCH_CHECK(q.dim() == 3, "q must be [bsz, n_q_heads, head_dim]");
  TORCH_CHECK(k.dim() == 4, "k must be [bsz, n_kv_heads, seq_len, head_dim]");
  TORCH_CHECK(scores.dim() == 3, "scores must be [bsz, n_q_heads, max_tokens]");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype mismatch");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32, "scores must be float32");
  TORCH_CHECK(q.size(0) == k.size(0), "q/k batch mismatch");
  TORCH_CHECK(q.size(0) == scores.size(0), "q/scores batch mismatch");
  TORCH_CHECK(q.size(1) == scores.size(1), "q/scores n_q_heads mismatch");
  TORCH_CHECK(q.size(2) == k.size(3), "q/k head_dim mismatch");
  TORCH_CHECK(n_q_per_kv > 0, "n_q_per_kv must be > 0");
  TORCH_CHECK((q.size(1) % n_q_per_kv) == 0, "n_q_heads must be divisible by n_q_per_kv");
  TORCH_CHECK((q.size(1) / n_q_per_kv) == k.size(1), "k n_kv_heads mismatch");
  TORCH_CHECK(token_begin >= 0 && token_count >= 0, "token_begin/token_count must be >= 0");
  TORCH_CHECK(token_begin + token_count <= k.size(2), "qk chunk exceeds k seq_len");
  TORCH_CHECK(token_begin + token_count <= scores.size(2), "qk chunk exceeds scores length");

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    launch_qk_scores_chunk<c_type>(q, k, scores, n_q_per_kv, token_begin, token_count);
    return true;
  });
}

torch::Tensor echo_decode_qk_scores_pagemax_chunk(
    const torch::Tensor &q,
    const torch::Tensor &k,
    torch::Tensor scores,
    int64_t n_q_per_kv,
    int64_t token_begin,
    int64_t page_size,
    int64_t page_count) {
  TORCH_CHECK(page_size > 0, "page_size must be > 0");
  TORCH_CHECK(page_count > 0, "page_count must be > 0");
  const int64_t token_count = page_size * page_count;
  echo_decode_qk_scores_chunk(q, k, scores, n_q_per_kv, token_begin, token_count);

  auto out_best = torch::empty(
      {q.size(0), q.size(1), page_count},
      torch::TensorOptions().dtype(torch::kInt32).device(q.device()));

  constexpr int THREADS = 128;
  const int64_t total = q.size(0) * q.size(1) * page_count;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  echo_page_argmax_from_scores_kernel<THREADS>
      <<<static_cast<uint32_t>(total), THREADS, 0, stream>>>(
          scores.data_ptr<float>(),
          out_best.data_ptr<int32_t>(),
          scores.stride(0),
          scores.stride(1),
          scores.stride(2),
          out_best.stride(0),
          out_best.stride(1),
          out_best.stride(2),
          q.size(1),
          token_begin,
          page_size,
          page_count);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "echo_page_argmax_from_scores_kernel launch failed: ",
      cudaGetErrorString(err));

  return out_best;
}

torch::Tensor echo_decode_qk_scores_pagemax_chunk_reduced(
    const torch::Tensor &q,
    const torch::Tensor &k,
    torch::Tensor scores,
    int64_t n_q_per_kv,
    int64_t token_begin,
    int64_t page_size,
    int64_t page_count) {
  auto out_qh = echo_decode_qk_scores_pagemax_chunk(
      q, k, scores, n_q_per_kv, token_begin, page_size, page_count);
  auto out_best = torch::empty(
      {q.size(0), page_count},
      torch::TensorOptions().dtype(torch::kInt32).device(q.device()));

  constexpr int THREADS = 128;
  const int64_t total = q.size(0) * page_count;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  echo_reduce_qh_best_mean_kernel<THREADS>
      <<<static_cast<uint32_t>(total), THREADS, 0, stream>>>(
          out_qh.data_ptr<int32_t>(),
          out_best.data_ptr<int32_t>(),
          out_qh.stride(0),
          out_qh.stride(1),
          out_qh.stride(2),
          out_best.stride(0),
          out_best.stride(1),
          q.size(1),
          page_count);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "echo_decode_qk_scores_pagemax_chunk_reduced launch failed: ",
      cudaGetErrorString(err));
  return out_best;
}

torch::Tensor echo_decode_qk_pagemax_chunk_only(
    const torch::Tensor &q,
    const torch::Tensor &k,
    int64_t n_q_per_kv,
    int64_t token_begin,
    int64_t page_size,
    int64_t page_count) {
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  TORCH_CHECK(q.dim() == 3, "q must be [bsz, n_q_heads, head_dim]");
  TORCH_CHECK(k.dim() == 4, "k must be [bsz, n_kv_heads, seq_len, head_dim]");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype mismatch");
  TORCH_CHECK(n_q_per_kv > 0, "n_q_per_kv must be > 0");
  TORCH_CHECK((q.size(1) % n_q_per_kv) == 0, "n_q_heads must be divisible by n_q_per_kv");
  TORCH_CHECK((q.size(1) / n_q_per_kv) == k.size(1), "k n_kv_heads mismatch");
  TORCH_CHECK(q.size(2) == k.size(3), "q/k head_dim mismatch");
  TORCH_CHECK(page_size > 0 && page_size <= 128, "page_size must be in [1, 128]");
  TORCH_CHECK(page_count > 0, "page_count must be > 0");
  TORCH_CHECK(token_begin >= 0, "token_begin must be >= 0");
  TORCH_CHECK(
      token_begin + page_size * page_count <= k.size(2),
      "chunk exceeds k seq_len");

  auto out_best = torch::empty(
      {q.size(0), q.size(1), page_count},
      torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
  constexpr int THREADS = 128;
  const int64_t total = q.size(0) * q.size(1) * page_count;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    echo_qk_pagemax_chunk_only_kernel<c_type, THREADS>
        <<<static_cast<uint32_t>(total), THREADS, 0, stream>>>(
            reinterpret_cast<const c_type*>(q.data_ptr()),
            reinterpret_cast<const c_type*>(k.data_ptr()),
            out_best.data_ptr<int32_t>(),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            out_best.stride(0),
            out_best.stride(1),
            out_best.stride(2),
            q.size(1),
            n_q_per_kv,
            q.size(2),
            token_begin,
            page_size,
            page_count);
    return true;
  });
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "echo_qk_pagemax_chunk_only_kernel launch failed: ",
      cudaGetErrorString(err));
  return out_best;
}

torch::Tensor echo_decode_qk_pagemax_chunk_only_reduced(
    const torch::Tensor &q,
    const torch::Tensor &k,
    int64_t n_q_per_kv,
    int64_t token_begin,
    int64_t page_size,
    int64_t page_count) {
  auto out_qh = echo_decode_qk_pagemax_chunk_only(
      q, k, n_q_per_kv, token_begin, page_size, page_count);
  auto out_best = torch::empty(
      {q.size(0), page_count},
      torch::TensorOptions().dtype(torch::kInt32).device(q.device()));

  constexpr int THREADS = 128;
  const int64_t total = q.size(0) * page_count;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  echo_reduce_qh_best_mean_kernel<THREADS>
      <<<static_cast<uint32_t>(total), THREADS, 0, stream>>>(
          out_qh.data_ptr<int32_t>(),
          out_best.data_ptr<int32_t>(),
          out_qh.stride(0),
          out_qh.stride(1),
          out_qh.stride(2),
          out_best.stride(0),
          out_best.stride(1),
          q.size(1),
          page_count);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "echo_reduce_qh_best_mean_kernel launch failed: ",
      cudaGetErrorString(err));
  return out_best;
}

torch::Tensor echo_decode_pv_from_scores_cuda(
    const torch::Tensor &scores,
    const torch::Tensor &v,
    int64_t n_q_per_kv,
    int64_t seq_len) {
  CHECK_CUDA(scores);
  CHECK_CUDA(v);
  TORCH_CHECK(scores.dim() == 3, "scores must be [bsz, n_q_heads, max_tokens]");
  TORCH_CHECK(v.dim() == 4, "v must be [bsz, n_kv_heads, seq_len, head_dim]");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32, "scores must be float32");
  TORCH_CHECK(n_q_per_kv > 0, "n_q_per_kv must be > 0");
  TORCH_CHECK(scores.size(0) == v.size(0), "scores/v batch mismatch");
  TORCH_CHECK(seq_len > 0 && seq_len <= v.size(2), "seq_len out of v range");
  TORCH_CHECK(seq_len <= scores.size(2), "seq_len out of scores range");
  const int64_t n_q_heads = scores.size(1);
  TORCH_CHECK((n_q_heads % n_q_per_kv) == 0, "n_q_heads must be divisible by n_q_per_kv");
  TORCH_CHECK((n_q_heads / n_q_per_kv) == v.size(1), "v n_kv_heads mismatch");

  auto out = torch::empty(
      {scores.size(0), n_q_heads, v.size(3)},
      scores.options().dtype(torch::kFloat32));
  if (scores.numel() == 0) {
    return out;
  }

  constexpr int THREADS = 128;
  const int64_t total = scores.size(0) * n_q_heads;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(v.scalar_type(), c_type, [&] {
    echo_pv_from_scores_kernel<c_type, THREADS>
        <<<static_cast<uint32_t>(total), THREADS, 0, stream>>>(
            scores.data_ptr<float>(),
            reinterpret_cast<const c_type*>(v.data_ptr()),
            out.data_ptr<float>(),
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            n_q_heads,
            n_q_per_kv,
            seq_len,
            v.size(3));
    return true;
  });
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "echo_pv_from_scores_kernel launch failed: ", cudaGetErrorString(err));
  return out;
}
