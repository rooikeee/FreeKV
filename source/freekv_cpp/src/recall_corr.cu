#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>

#include "pytorch_extension_utils.h"

torch::Tensor alloc_managed_bool(int64_t rows, int64_t cols) {
  TORCH_CHECK(rows > 0 && cols > 0, "rows and cols must be positive");
  bool* ptr = nullptr;
  cudaError_t err = cudaMallocManaged(
      reinterpret_cast<void**>(&ptr), rows * cols * sizeof(bool), cudaMemAttachGlobal);
  TORCH_CHECK(err == cudaSuccess, "cudaMallocManaged failed: ", cudaGetErrorString(err));
  auto options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
  auto t = torch::from_blob(
      ptr, {rows, cols}, [](void* p) { cudaFree(p); }, options);
  t.fill_(false);
  return t;
}

torch::Tensor alloc_managed_bool_scalar() {
  int32_t* ptr = nullptr;
  cudaError_t err = cudaMallocManaged(
      reinterpret_cast<void**>(&ptr), sizeof(int32_t), cudaMemAttachGlobal);
  TORCH_CHECK(err == cudaSuccess, "cudaMallocManaged failed: ", cudaGetErrorString(err));
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  auto t = torch::from_blob(
      ptr, {1}, [](void* p) { cudaFree(p); }, options);
  t.fill_(0);
  return t;
}

template <typename scalar_t>
__global__ void corr_flags_kernel(
    const scalar_t* __restrict__ query_states,   // [bsz, 1, num_heads, head_dim]
    const scalar_t* __restrict__ last_step_q,    // [bsz, 1, num_heads, head_dim]
    bool* __restrict__ to_corr,                  // [bsz, n_kv_heads]
    int32_t* __restrict__ need_corr,             // [1], int32 flag
    int64_t to_corr_stride0,
    int64_t to_corr_stride1,
    int32_t bsz,
    int32_t num_heads,
    int32_t head_dim,
    int32_t n_kv_heads,
    float corr) {
  constexpr int BLOCK_THREADS = 128;
  __shared__ float s_dot[BLOCK_THREADS];
  __shared__ float s_norm_q[BLOCK_THREADS];
  __shared__ float s_norm_l[BLOCK_THREADS];

  const int idx = blockIdx.x;
  const int total = bsz * n_kv_heads;
  if (idx >= total) {
    return;
  }
  const int b = idx / n_kv_heads;
  const int kv = idx % n_kv_heads;
  const int group_size = num_heads / n_kv_heads;

  float sim_sum = 0.0f;
  for (int g = 0; g < group_size; ++g) {
    const int h = kv * group_size + g;
    const int64_t base = (static_cast<int64_t>(b) * num_heads + h) * head_dim;
    float dot_local = 0.0f;
    float norm_q_local = 0.0f;
    float norm_l_local = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      const float q = static_cast<float>(query_states[base + d]);
      const float l = static_cast<float>(last_step_q[base + d]);
      dot_local += q * l;
      norm_q_local += q * q;
      norm_l_local += l * l;
    }
    s_dot[threadIdx.x] = dot_local;
    s_norm_q[threadIdx.x] = norm_q_local;
    s_norm_l[threadIdx.x] = norm_l_local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        s_dot[threadIdx.x] += s_dot[threadIdx.x + stride];
        s_norm_q[threadIdx.x] += s_norm_q[threadIdx.x + stride];
        s_norm_l[threadIdx.x] += s_norm_l[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      const float denom = sqrtf(s_norm_q[0] * s_norm_l[0]) + 1e-12f;
      sim_sum += s_dot[0] / denom;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const float sim_mean = sim_sum / static_cast<float>(group_size);
    const bool flag = sim_mean < corr;
    to_corr[b * to_corr_stride0 + kv * to_corr_stride1] = flag;
    if (flag) {
      atomicOr(need_corr, 1);
    }
  }
}

static int32_t* g_need_corr_device = nullptr;   // device int32
static int32_t* g_need_corr_pinned = nullptr;   // pinned-host int32

static void ensure_need_corr_buffers() {
  if (g_need_corr_device == nullptr) {
    cudaError_t err;
    err = cudaMalloc(reinterpret_cast<void**>(&g_need_corr_device), sizeof(int32_t));
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc failed: ", cudaGetErrorString(err));
    err = cudaMemset(g_need_corr_device, 0, sizeof(int32_t));
    TORCH_CHECK(err == cudaSuccess, "cudaMemset failed: ", cudaGetErrorString(err));
    err = cudaMallocHost(reinterpret_cast<void**>(&g_need_corr_pinned), sizeof(int32_t));
    TORCH_CHECK(err == cudaSuccess, "cudaMallocHost failed: ", cudaGetErrorString(err));
    *g_need_corr_pinned = 0;
  }
}

bool get_corr_managed_cuda(
    const torch::Tensor &query_states,
    const torch::Tensor &last_step_q,
    int64_t n_kv_heads,
    float corr,
    torch::Tensor to_corr_managed) {
  CHECK_INPUT(query_states);
  CHECK_INPUT(last_step_q);
  CHECK_DIM(4, query_states);
  CHECK_DIM(4, last_step_q);
  CHECK_EQ(query_states.scalar_type(), last_step_q.scalar_type());
  CHECK_EQ(query_states.size(1), 1);
  CHECK_EQ(last_step_q.size(1), 1);
  CHECK_EQ(query_states.size(0), last_step_q.size(0));
  CHECK_EQ(query_states.size(2), last_step_q.size(2));
  CHECK_EQ(query_states.size(3), last_step_q.size(3));
  CHECK_EQ(query_states.size(2) % n_kv_heads, 0);

  CHECK_DIM(2, to_corr_managed);
  CHECK_CONTIGUOUS(to_corr_managed);
  CHECK_EQ(to_corr_managed.scalar_type(), torch::kBool);
  TORCH_CHECK(!to_corr_managed.is_cuda(), "to_corr_managed must be a CPU tensor");

  const int32_t bsz = query_states.size(0);
  const int32_t num_heads = query_states.size(2);
  const int32_t head_dim = query_states.size(3);
  CHECK_EQ(to_corr_managed.size(0), bsz);
  CHECK_EQ(to_corr_managed.size(1), n_kv_heads);
  const int64_t to_corr_stride0 = to_corr_managed.stride(0);
  const int64_t to_corr_stride1 = to_corr_managed.stride(1);
  TORCH_CHECK(to_corr_stride0 > 0 && to_corr_stride1 > 0,
              "to_corr_managed must have positive strides");

  const int total = bsz * n_kv_heads;
  if (total == 0) {
    return false;
  }

  ensure_need_corr_buffers();

  cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

  // Reset on device — purely GPU-side, no page faults
  cudaError_t memset_err = cudaMemsetAsync(
      g_need_corr_device, 0, sizeof(int32_t), curr_stream);
  TORCH_CHECK(memset_err == cudaSuccess, "cudaMemsetAsync failed: ",
              cudaGetErrorString(memset_err));

  constexpr int threads = 128;
  const int blocks = total;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      query_states.scalar_type(), "corr_flags_kernel", [&] {
        corr_flags_kernel<scalar_t><<<blocks, threads, 0, curr_stream>>>(
            query_states.data_ptr<scalar_t>(),
            last_step_q.data_ptr<scalar_t>(),
            to_corr_managed.data_ptr<bool>(),
            g_need_corr_device,
            to_corr_stride0,
            to_corr_stride1,
            bsz,
            num_heads,
            head_dim,
            n_kv_heads,
            corr);
      });
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "get_corr_managed_cuda kernel launch failed: ",
              cudaGetErrorString(err));

  // 4-byte D2H copy to pinned host — no page faults
  err = cudaMemcpyAsync(
      g_need_corr_pinned, g_need_corr_device,
      sizeof(int32_t), cudaMemcpyDeviceToHost, curr_stream);
  TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync D2H failed: ",
              cudaGetErrorString(err));

  cudaError_t sync_err = cudaStreamSynchronize(curr_stream);
  TORCH_CHECK(sync_err == cudaSuccess, "cudaStreamSynchronize failed: ",
              cudaGetErrorString(sync_err));

  return *g_need_corr_pinned != 0;
}
