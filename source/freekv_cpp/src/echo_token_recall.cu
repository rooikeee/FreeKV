#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>

#include "pytorch_extension_utils.h"

void recall_tokens_linear(
    const torch::Tensor &token_starts,
    const torch::Tensor &cpu_kv_linear,
    torch::Tensor gpu_mid_kv,
    int64_t valid_tokens
) {
  CHECK_CONTIGUOUS(token_starts);
  TORCH_CHECK(token_starts.dim() == 2, "token_starts must be 2D");
  TORCH_CHECK(token_starts.scalar_type() == torch::kInt32 || token_starts.scalar_type() == torch::kInt64,
              "token_starts must be int32 or int64");

  TORCH_CHECK(!cpu_kv_linear.is_cuda(), "cpu_kv_linear must be CPU tensor");
  CHECK_CONTIGUOUS(cpu_kv_linear);
  TORCH_CHECK(cpu_kv_linear.is_pinned(), "cpu_kv_linear must be pinned CPU tensor");
  TORCH_CHECK(cpu_kv_linear.dim() == 5, "cpu_kv_linear must be [bsz, max_tokens, 2, n_kv_heads, head_dim]");

  CHECK_INPUT(gpu_mid_kv);
  TORCH_CHECK(gpu_mid_kv.dim() == 5, "gpu_mid_kv must be [bsz, mid_tokens, 2, n_kv_heads, head_dim]");

  const int64_t starts_bsz = token_starts.size(0);
  const int64_t n_pages = token_starts.size(1);
  const int64_t bsz = gpu_mid_kv.size(0);
  const int64_t mid_tokens = gpu_mid_kv.size(1);
  TORCH_CHECK(n_pages > 0, "token_starts second dim (n_pages) must be > 0");
  TORCH_CHECK(mid_tokens % n_pages == 0, "mid_tokens must be divisible by n_pages");
  const int64_t page_size = mid_tokens / n_pages;

  TORCH_CHECK(cpu_kv_linear.size(0) == bsz, "cpu_kv_linear batch must match gpu_mid_kv batch");
  TORCH_CHECK(cpu_kv_linear.size(2) == gpu_mid_kv.size(2), "kv dim mismatch");
  TORCH_CHECK(cpu_kv_linear.size(3) == gpu_mid_kv.size(3), "n_kv_heads mismatch");
  TORCH_CHECK(cpu_kv_linear.size(4) == gpu_mid_kv.size(4), "head_dim mismatch");
  TORCH_CHECK(starts_bsz == 1 || starts_bsz == bsz, "token_starts batch must be 1 or bsz");
  TORCH_CHECK(valid_tokens >= 0 && valid_tokens <= cpu_kv_linear.size(1),
              "valid_tokens out of range");

  const int64_t cpu_stride_tok = cpu_kv_linear.stride(1);
  const int64_t gpu_stride_tok = gpu_mid_kv.stride(1);
  TORCH_CHECK(cpu_stride_tok == gpu_stride_tok,
              "token stride mismatch between cpu_kv_linear and gpu_mid_kv");

  auto starts_cpu = token_starts.is_cuda() ? token_starts.to(torch::kCPU) : token_starts;
  if (starts_cpu.scalar_type() != torch::kInt32) {
    starts_cpu = starts_cpu.to(torch::kInt32);
  }
  starts_cpu = starts_cpu.contiguous();
  const int32_t* starts_ptr = starts_cpu.data_ptr<int32_t>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(gpu_mid_kv.scalar_type(), c_type, [&] {
    using scalar_t = c_type;
    scalar_t *dst_base = reinterpret_cast<scalar_t *>(gpu_mid_kv.data_ptr());
    scalar_t *src_base = reinterpret_cast<scalar_t *>(cpu_kv_linear.data_ptr());
    const size_t elem_size = sizeof(scalar_t);
    const size_t page_bytes = static_cast<size_t>(page_size * cpu_stride_tok) * elem_size;
    const int64_t src_stride_b = cpu_kv_linear.stride(0);
    const int64_t dst_stride_b = gpu_mid_kv.stride(0);

    for (int64_t b = 0; b < bsz; ++b) {
      const int64_t sb = (starts_bsz == 1) ? 0 : b;
      int64_t p = 0;
      while (p < n_pages) {
        const int64_t p0 = p;
        const int64_t s0 = static_cast<int64_t>(starts_ptr[sb * n_pages + p0]);
        TORCH_CHECK(s0 >= 0 && s0 + page_size <= valid_tokens,
                    "token start out of valid range: start=", s0,
                    ", page_size=", page_size, ", valid_tokens=", valid_tokens);
        int64_t run_pages = 1;
        while (p0 + run_pages < n_pages) {
          const int64_t sn = static_cast<int64_t>(starts_ptr[sb * n_pages + p0 + run_pages]);
          TORCH_CHECK(sn >= 0 && sn + page_size <= valid_tokens,
                      "token start out of valid range: start=", sn,
                      ", page_size=", page_size, ", valid_tokens=", valid_tokens);
          if (sn != s0 + run_pages * page_size) {
            break;
          }
          ++run_pages;
        }
        scalar_t *src_ptr = src_base + b * src_stride_b + s0 * cpu_stride_tok;
        scalar_t *dst_ptr = dst_base + b * dst_stride_b + (p0 * page_size) * gpu_stride_tok;
        const size_t run_bytes = page_bytes * static_cast<size_t>(run_pages);
        cudaError_t err = cudaMemcpyAsync(
            dst_ptr,
            src_ptr,
            run_bytes,
            cudaMemcpyHostToDevice,
            stream);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
        p += run_pages;
      }
    }
    return true;
  });
}
