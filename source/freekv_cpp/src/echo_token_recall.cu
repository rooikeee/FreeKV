#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>

#include <cstdint>
#include <cstdlib>

#include "pytorch_extension_utils.h"

namespace {

torch::Tensor to_cpu_i32_starts(const torch::Tensor& starts, const char* name) {
  CHECK_CONTIGUOUS(starts);
  TORCH_CHECK(starts.dim() == 2, name, " must be 2D");
  TORCH_CHECK(
      starts.scalar_type() == torch::kInt32 || starts.scalar_type() == torch::kInt64,
      name,
      " must be int32 or int64");
  auto cpu = starts.is_cuda() ? starts.to(torch::kCPU) : starts;
  if (cpu.scalar_type() != torch::kInt32) {
    cpu = cpu.to(torch::kInt32);
  }
  return cpu.contiguous();
}

void check_starts_bounds(
    const int32_t* starts_ptr,
    int64_t starts_bsz,
    int64_t n_pages,
    int64_t page_size,
    int64_t valid_tokens,
    const char* name) {
  for (int64_t sb = 0; sb < starts_bsz; ++sb) {
    for (int64_t p = 0; p < n_pages; ++p) {
      const int64_t s = static_cast<int64_t>(starts_ptr[sb * n_pages + p]);
      TORCH_CHECK(
          s >= 0 && s + page_size <= valid_tokens,
          name,
          " out of valid range: start=",
          s,
          ", page_size=",
          page_size,
          ", valid_tokens=",
          valid_tokens);
    }
  }
}

bool recall_bounds_check_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char* env = std::getenv("FREEKV_RECALL_CHECK_BOUNDS");
    enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
  }
  return enabled != 0;
}

template <typename scalar_t>
inline void memcpy_async_checked(
    scalar_t* dst,
    const scalar_t* src,
    size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream,
    const char* tag) {
  if (bytes == 0) {
    return;
  }
  cudaError_t err = cudaMemcpyAsync(dst, src, bytes, kind, stream);
  TORCH_CHECK(err == cudaSuccess, tag, " cudaMemcpyAsync failed: ", cudaGetErrorString(err));
}

inline void memcpy2d_async_checked(
    void* dst,
    size_t dst_pitch,
    const void* src,
    size_t src_pitch,
    size_t width,
    size_t height,
    cudaMemcpyKind kind,
    cudaStream_t stream,
    const char* tag) {
  if (width == 0 || height == 0) {
    return;
  }
  cudaError_t err =
      cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height, kind, stream);
  TORCH_CHECK(err == cudaSuccess, tag, " cudaMemcpy2DAsync failed: ", cudaGetErrorString(err));
}

}  // namespace

void recall_tokens_linear(
    const torch::Tensor &token_starts,
    const torch::Tensor &cpu_kv_linear,
    torch::Tensor gpu_mid_kv,
    int64_t valid_tokens
) {
  TORCH_CHECK(!cpu_kv_linear.is_cuda(), "cpu_kv_linear must be CPU tensor");
  CHECK_CONTIGUOUS(cpu_kv_linear);
  TORCH_CHECK(cpu_kv_linear.is_pinned(), "cpu_kv_linear must be pinned CPU tensor");
  TORCH_CHECK(cpu_kv_linear.dim() == 5, "cpu_kv_linear must be [bsz, max_tokens, 2, n_kv_heads, head_dim]");

  CHECK_INPUT(gpu_mid_kv);
  TORCH_CHECK(gpu_mid_kv.dim() == 5, "gpu_mid_kv must be [bsz, mid_tokens, 2, n_kv_heads, head_dim]");

  auto starts_cpu = to_cpu_i32_starts(token_starts, "token_starts");
  const int64_t starts_bsz = starts_cpu.size(0);
  const int64_t n_pages = starts_cpu.size(1);
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
  TORCH_CHECK(
      valid_tokens >= 0 && valid_tokens <= cpu_kv_linear.size(1),
      "valid_tokens out of range");

  const int64_t cpu_stride_tok = cpu_kv_linear.stride(1);
  const int64_t gpu_stride_tok = gpu_mid_kv.stride(1);
  TORCH_CHECK(
      cpu_stride_tok == gpu_stride_tok,
      "token stride mismatch between cpu_kv_linear and gpu_mid_kv");

  const int32_t* starts_ptr = starts_cpu.data_ptr<int32_t>();
  if (recall_bounds_check_enabled()) {
    check_starts_bounds(starts_ptr, starts_bsz, n_pages, page_size, valid_tokens, "token_starts");
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(gpu_mid_kv.scalar_type(), c_type, [&] {
    using scalar_t = c_type;
    scalar_t* dst_base = reinterpret_cast<scalar_t*>(gpu_mid_kv.data_ptr());
    scalar_t* src_base = reinterpret_cast<scalar_t*>(cpu_kv_linear.data_ptr());

    const size_t token_bytes = static_cast<size_t>(gpu_stride_tok) * sizeof(scalar_t);
    const size_t page_bytes = static_cast<size_t>(page_size) * token_bytes;
    const int64_t src_stride_b = cpu_kv_linear.stride(0);
    const int64_t dst_stride_b = gpu_mid_kv.stride(0);

    for (int64_t b = 0; b < bsz; ++b) {
      const int64_t sb = (starts_bsz == 1) ? 0 : b;
      int64_t p = 0;
      while (p < n_pages) {
        const int64_t p0 = p;
        const int64_t s0 = static_cast<int64_t>(starts_ptr[sb * n_pages + p0]);
        int64_t run_pages = 1;
        while (p0 + run_pages < n_pages) {
          const int64_t sn = static_cast<int64_t>(starts_ptr[sb * n_pages + p0 + run_pages]);
          if (sn != s0 + run_pages * page_size) {
            break;
          }
          ++run_pages;
        }

        scalar_t* src_ptr = src_base + b * src_stride_b + s0 * cpu_stride_tok;
        scalar_t* dst_ptr = dst_base + b * dst_stride_b + (p0 * page_size) * gpu_stride_tok;
        const size_t run_bytes = page_bytes * static_cast<size_t>(run_pages);
        memcpy_async_checked(dst_ptr, src_ptr, run_bytes, cudaMemcpyHostToDevice, stream, "recall_tokens_linear");
        p += run_pages;
      }
    }
    return true;
  });
}

void recall_tokens_delta_linear(
    const torch::Tensor &token_starts,
    const torch::Tensor &prev_token_starts,
    const torch::Tensor &cpu_kv_linear,
    const torch::Tensor &gpu_prev_mid_kv,
    torch::Tensor gpu_mid_kv,
    int64_t valid_tokens
) {
  TORCH_CHECK(!cpu_kv_linear.is_cuda(), "cpu_kv_linear must be CPU tensor");
  CHECK_CONTIGUOUS(cpu_kv_linear);
  TORCH_CHECK(cpu_kv_linear.is_pinned(), "cpu_kv_linear must be pinned CPU tensor");
  TORCH_CHECK(cpu_kv_linear.dim() == 5, "cpu_kv_linear must be [bsz, max_tokens, 2, n_kv_heads, head_dim]");

  CHECK_INPUT(gpu_prev_mid_kv);
  CHECK_INPUT(gpu_mid_kv);
  TORCH_CHECK(gpu_prev_mid_kv.dim() == 5, "gpu_prev_mid_kv must be [bsz, mid_tokens, 2, n_kv_heads, head_dim]");
  TORCH_CHECK(gpu_mid_kv.dim() == 5, "gpu_mid_kv must be [bsz, mid_tokens, 2, n_kv_heads, head_dim]");
  TORCH_CHECK(gpu_prev_mid_kv.sizes() == gpu_mid_kv.sizes(), "gpu_prev_mid_kv and gpu_mid_kv must have identical shape");
  TORCH_CHECK(gpu_prev_mid_kv.scalar_type() == gpu_mid_kv.scalar_type(), "gpu_prev_mid_kv and gpu_mid_kv dtype mismatch");

  auto starts_cpu = to_cpu_i32_starts(token_starts, "token_starts");
  auto prev_starts_cpu = to_cpu_i32_starts(prev_token_starts, "prev_token_starts");

  const int64_t starts_bsz = starts_cpu.size(0);
  const int64_t prev_starts_bsz = prev_starts_cpu.size(0);
  const int64_t n_pages = starts_cpu.size(1);
  TORCH_CHECK(prev_starts_cpu.size(1) == n_pages, "token_starts and prev_token_starts must have same n_pages");

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
  TORCH_CHECK(prev_starts_bsz == 1 || prev_starts_bsz == bsz, "prev_token_starts batch must be 1 or bsz");

  TORCH_CHECK(
      valid_tokens >= 0 && valid_tokens <= cpu_kv_linear.size(1),
      "valid_tokens out of range");

  const int64_t cpu_stride_tok = cpu_kv_linear.stride(1);
  const int64_t prev_stride_tok = gpu_prev_mid_kv.stride(1);
  const int64_t dst_stride_tok = gpu_mid_kv.stride(1);
  TORCH_CHECK(cpu_stride_tok == dst_stride_tok, "token stride mismatch between cpu_kv_linear and gpu_mid_kv");
  TORCH_CHECK(prev_stride_tok == dst_stride_tok, "token stride mismatch between gpu_prev_mid_kv and gpu_mid_kv");

  const int32_t* starts_ptr = starts_cpu.data_ptr<int32_t>();
  const int32_t* prev_starts_ptr = prev_starts_cpu.data_ptr<int32_t>();
  if (recall_bounds_check_enabled()) {
    check_starts_bounds(starts_ptr, starts_bsz, n_pages, page_size, valid_tokens, "token_starts");
    check_starts_bounds(prev_starts_ptr, prev_starts_bsz, n_pages, page_size, valid_tokens, "prev_token_starts");
  }

  const bool same_buffer = gpu_prev_mid_kv.data_ptr() == gpu_mid_kv.data_ptr();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(gpu_mid_kv.scalar_type(), c_type, [&] {
    using scalar_t = c_type;
    scalar_t* dst_base = reinterpret_cast<scalar_t*>(gpu_mid_kv.data_ptr());
    scalar_t* prev_base = reinterpret_cast<scalar_t*>(gpu_prev_mid_kv.data_ptr());
    scalar_t* src_base = reinterpret_cast<scalar_t*>(cpu_kv_linear.data_ptr());

    const int64_t src_stride_b = cpu_kv_linear.stride(0);
    const int64_t prev_stride_b = gpu_prev_mid_kv.stride(0);
    const int64_t dst_stride_b = gpu_mid_kv.stride(0);

    const size_t token_bytes = static_cast<size_t>(dst_stride_tok) * sizeof(scalar_t);
    const size_t page_bytes = static_cast<size_t>(page_size) * token_bytes;
    const size_t page_pitch_bytes = static_cast<size_t>(page_size) * token_bytes;

    for (int64_t b = 0; b < bsz; ++b) {
      const int64_t sb_new = (starts_bsz == 1) ? 0 : b;
      const int64_t sb_old = (prev_starts_bsz == 1) ? 0 : b;

      if (!same_buffer) {
        scalar_t* dst_batch = dst_base + b * dst_stride_b;
        scalar_t* prev_batch = prev_base + b * prev_stride_b;
        memcpy_async_checked(
            dst_batch,
            prev_batch,
            page_bytes * static_cast<size_t>(n_pages),
            cudaMemcpyDeviceToDevice,
            stream,
            "recall_tokens_delta_linear(d2d_full_prev)");

        int64_t p2 = 0;
        while (p2 < n_pages) {
          const int64_t s_new =
              static_cast<int64_t>(starts_ptr[sb_new * n_pages + p2]);
          const int64_t s_old =
              static_cast<int64_t>(prev_starts_ptr[sb_old * n_pages + p2]);
          const int64_t shift = s_new - s_old;
          if (shift == 0) {
            ++p2;
            continue;
          }

          scalar_t* dst_page =
              dst_batch + (p2 * page_size) * dst_stride_tok;

          if (shift >= page_size || shift <= -page_size) {
            int64_t run_pages = 1;
            while (p2 + run_pages < n_pages) {
              const int64_t sn = static_cast<int64_t>(
                  starts_ptr[sb_new * n_pages + p2 + run_pages]);
              const int64_t so = static_cast<int64_t>(
                  prev_starts_ptr[sb_old * n_pages + p2 + run_pages]);
              const int64_t sh = sn - so;
              if (!(sh >= page_size || sh <= -page_size)) {
                break;
              }
              if (sn != s_new + run_pages * page_size) {
                break;
              }
              ++run_pages;
            }
            scalar_t* src_ptr = src_base + b * src_stride_b + s_new * cpu_stride_tok;
            memcpy_async_checked(
                dst_page,
                src_ptr,
                page_bytes * static_cast<size_t>(run_pages),
                cudaMemcpyHostToDevice,
                stream,
                "recall_tokens_delta_linear(h2d_full_patch)");
            p2 += run_pages;
            continue;
          }

          int64_t run_pages = 1;
          while (p2 + run_pages < n_pages) {
            const int64_t sn = static_cast<int64_t>(
                starts_ptr[sb_new * n_pages + p2 + run_pages]);
            const int64_t so = static_cast<int64_t>(
                prev_starts_ptr[sb_old * n_pages + p2 + run_pages]);
            const int64_t sh = sn - so;
            if (sh != shift) {
              break;
            }
            if (sn != s_new + run_pages * page_size) {
              break;
            }
            ++run_pages;
          }

          if (shift > 0) {
            const int64_t overlap = page_size - shift;
            scalar_t* h2d_src = src_base + b * src_stride_b + (s_new + overlap) * cpu_stride_tok;
            scalar_t* h2d_dst = dst_page + overlap * dst_stride_tok;
            memcpy2d_async_checked(
                h2d_dst,
                page_pitch_bytes,
                h2d_src,
                page_pitch_bytes,
                static_cast<size_t>(shift) * token_bytes,
                static_cast<size_t>(run_pages),
                cudaMemcpyHostToDevice,
                stream,
                "recall_tokens_delta_linear(h2d_tail_patch)");
          } else {
            const int64_t shift_abs = -shift;
            scalar_t* h2d_src = src_base + b * src_stride_b + s_new * cpu_stride_tok;
            memcpy2d_async_checked(
                dst_page,
                page_pitch_bytes,
                h2d_src,
                page_pitch_bytes,
                static_cast<size_t>(shift_abs) * token_bytes,
                static_cast<size_t>(run_pages),
                cudaMemcpyHostToDevice,
                stream,
                "recall_tokens_delta_linear(h2d_head_patch)");
          }
          p2 += run_pages;
        }
        continue;
      }

      int64_t p = 0;
      while (p < n_pages) {
        const int64_t s_new = static_cast<int64_t>(starts_ptr[sb_new * n_pages + p]);
        const int64_t s_old = static_cast<int64_t>(prev_starts_ptr[sb_old * n_pages + p]);
        const int64_t shift = s_new - s_old;

        scalar_t* dst_page = dst_base + b * dst_stride_b + (p * page_size) * dst_stride_tok;
        scalar_t* prev_page = prev_base + b * prev_stride_b + (p * page_size) * prev_stride_tok;

        if (same_buffer) {
          if (shift == 0) {
            int64_t run_pages = 1;
            while (p + run_pages < n_pages) {
              const int64_t sn =
                  static_cast<int64_t>(starts_ptr[sb_new * n_pages + p + run_pages]);
              const int64_t so =
                  static_cast<int64_t>(prev_starts_ptr[sb_old * n_pages + p + run_pages]);
              if (sn != so) {
                break;
              }
              ++run_pages;
            }
            p += run_pages;
            continue;
          }

          if (shift >= page_size || shift <= -page_size) {
            int64_t run_pages = 1;
            while (p + run_pages < n_pages) {
              const int64_t sn =
                  static_cast<int64_t>(starts_ptr[sb_new * n_pages + p + run_pages]);
              const int64_t so =
                  static_cast<int64_t>(prev_starts_ptr[sb_old * n_pages + p + run_pages]);
              const int64_t sh = sn - so;
              if (!(sh >= page_size || sh <= -page_size)) {
                break;
              }
              if (sn != s_new + run_pages * page_size) {
                break;
              }
              ++run_pages;
            }
            scalar_t* src_ptr = src_base + b * src_stride_b + s_new * cpu_stride_tok;
            memcpy_async_checked(
                dst_page,
                src_ptr,
                page_bytes * static_cast<size_t>(run_pages),
                cudaMemcpyHostToDevice,
                stream,
                "recall_tokens_delta_linear(same_buffer_full)");
            p += run_pages;
            continue;
          }

          int64_t run_pages = 1;
          while (p + run_pages < n_pages) {
            const int64_t sn =
                static_cast<int64_t>(starts_ptr[sb_new * n_pages + p + run_pages]);
            const int64_t so =
                static_cast<int64_t>(prev_starts_ptr[sb_old * n_pages + p + run_pages]);
            const int64_t sh = sn - so;
            if (sh != shift) {
              break;
            }
            if (sn != s_new + run_pages * page_size) {
              break;
            }
            ++run_pages;
          }

          if (shift > 0) {
            const int64_t overlap = page_size - shift;
            scalar_t* h2d_src = src_base + b * src_stride_b + (s_new + overlap) * cpu_stride_tok;
            scalar_t* h2d_dst = dst_page + overlap * dst_stride_tok;
            memcpy2d_async_checked(
                h2d_dst,
                page_pitch_bytes,
                h2d_src,
                page_pitch_bytes,
                static_cast<size_t>(shift) * token_bytes,
                static_cast<size_t>(run_pages),
                cudaMemcpyHostToDevice,
                stream,
                "recall_tokens_delta_linear(same_buffer_h2d_tail)");
          } else {
            const int64_t shift_abs = -shift;
            scalar_t* h2d_src = src_base + b * src_stride_b + s_new * cpu_stride_tok;
            memcpy2d_async_checked(
                dst_page,
                page_pitch_bytes,
                h2d_src,
                page_pitch_bytes,
                static_cast<size_t>(shift_abs) * token_bytes,
                static_cast<size_t>(run_pages),
                cudaMemcpyHostToDevice,
                stream,
                "recall_tokens_delta_linear(same_buffer_h2d_head)");
          }
          p += run_pages;
          continue;
        }

        if (shift == 0) {
          int64_t run_pages = 1;
          while (p + run_pages < n_pages) {
            const int64_t sn = static_cast<int64_t>(starts_ptr[sb_new * n_pages + p + run_pages]);
            const int64_t so = static_cast<int64_t>(prev_starts_ptr[sb_old * n_pages + p + run_pages]);
            if (sn != so) {
              break;
            }
            ++run_pages;
          }
          memcpy_async_checked(
              dst_page,
              prev_page,
              page_bytes * static_cast<size_t>(run_pages),
              cudaMemcpyDeviceToDevice,
              stream,
              "recall_tokens_delta_linear(d2d_same)");
          p += run_pages;
          continue;
        }

        if (shift >= page_size || shift <= -page_size) {
          int64_t run_pages = 1;
          while (p + run_pages < n_pages) {
            const int64_t sn = static_cast<int64_t>(starts_ptr[sb_new * n_pages + p + run_pages]);
            const int64_t so = static_cast<int64_t>(prev_starts_ptr[sb_old * n_pages + p + run_pages]);
            const int64_t sh = sn - so;
            if (!(sh >= page_size || sh <= -page_size)) {
              break;
            }
            if (sn != s_new + run_pages * page_size) {
              break;
            }
            ++run_pages;
          }
          scalar_t* src_ptr = src_base + b * src_stride_b + s_new * cpu_stride_tok;
          memcpy_async_checked(
              dst_page,
              src_ptr,
              page_bytes * static_cast<size_t>(run_pages),
              cudaMemcpyHostToDevice,
              stream,
              "recall_tokens_delta_linear(h2d_full)");
          p += run_pages;
          continue;
        }

        int64_t run_pages = 1;
        while (p + run_pages < n_pages) {
          const int64_t sn = static_cast<int64_t>(starts_ptr[sb_new * n_pages + p + run_pages]);
          const int64_t so =
              static_cast<int64_t>(prev_starts_ptr[sb_old * n_pages + p + run_pages]);
          const int64_t sh = sn - so;
          if (sh != shift) {
            break;
          }
          if (sn != s_new + run_pages * page_size) {
            break;
          }
          if (so != s_old + run_pages * page_size) {
            break;
          }
          ++run_pages;
        }

        if (shift > 0) {
          const int64_t overlap = page_size - shift;
          scalar_t* d2d_src = prev_page + shift * prev_stride_tok;
          memcpy2d_async_checked(
              dst_page,
              page_pitch_bytes,
              d2d_src,
              page_pitch_bytes,
              static_cast<size_t>(overlap) * token_bytes,
              static_cast<size_t>(run_pages),
              cudaMemcpyDeviceToDevice,
              stream,
              "recall_tokens_delta_linear(d2d_tail)");

          scalar_t* h2d_src = src_base + b * src_stride_b + (s_new + overlap) * cpu_stride_tok;
          scalar_t* h2d_dst = dst_page + overlap * dst_stride_tok;
          memcpy2d_async_checked(
              h2d_dst,
              page_pitch_bytes,
              h2d_src,
              page_pitch_bytes,
              static_cast<size_t>(shift) * token_bytes,
              static_cast<size_t>(run_pages),
              cudaMemcpyHostToDevice,
              stream,
              "recall_tokens_delta_linear(h2d_tail)");
        } else {
          const int64_t shift_abs = -shift;
          const int64_t overlap = page_size - shift_abs;
          scalar_t* d2d_dst = dst_page + shift_abs * dst_stride_tok;
          memcpy2d_async_checked(
              d2d_dst,
              page_pitch_bytes,
              prev_page,
              page_pitch_bytes,
              static_cast<size_t>(overlap) * token_bytes,
              static_cast<size_t>(run_pages),
              cudaMemcpyDeviceToDevice,
              stream,
              "recall_tokens_delta_linear(d2d_head)");

          scalar_t* h2d_src = src_base + b * src_stride_b + s_new * cpu_stride_tok;
          memcpy2d_async_checked(
              dst_page,
              page_pitch_bytes,
              h2d_src,
              page_pitch_bytes,
              static_cast<size_t>(shift_abs) * token_bytes,
              static_cast<size_t>(run_pages),
              cudaMemcpyHostToDevice,
              stream,
              "recall_tokens_delta_linear(h2d_head)");
        }

        p += run_pages;
      }
    }
    return true;
  });
}

void recall_tokens_linear_partial(
    const torch::Tensor &token_starts,
    const torch::Tensor &cpu_kv_linear,
    torch::Tensor gpu_mid_kv,
    int64_t valid_tokens,
    int64_t page_begin,
    int64_t page_count
) {
  TORCH_CHECK(!cpu_kv_linear.is_cuda(), "cpu_kv_linear must be CPU tensor");
  CHECK_CONTIGUOUS(cpu_kv_linear);
  TORCH_CHECK(cpu_kv_linear.is_pinned(), "cpu_kv_linear must be pinned CPU tensor");
  TORCH_CHECK(cpu_kv_linear.dim() == 5, "cpu_kv_linear must be [bsz, max_tokens, 2, n_kv_heads, head_dim]");
  CHECK_INPUT(gpu_mid_kv);
  TORCH_CHECK(gpu_mid_kv.dim() == 5, "gpu_mid_kv must be [bsz, mid_tokens, 2, n_kv_heads, head_dim]");

  auto starts_cpu = to_cpu_i32_starts(token_starts, "token_starts");
  const int64_t starts_bsz = starts_cpu.size(0);
  const int64_t n_pages_total = starts_cpu.size(1);
  const int64_t bsz = gpu_mid_kv.size(0);
  const int64_t mid_tokens = gpu_mid_kv.size(1);

  TORCH_CHECK(page_begin >= 0, "page_begin must be >= 0");
  TORCH_CHECK(page_count >= 0, "page_count must be >= 0");
  TORCH_CHECK(
      page_begin + page_count <= n_pages_total,
      "page_begin + page_count exceeds token_starts second dim");
  if (page_count == 0) {
    return;
  }
  TORCH_CHECK(n_pages_total > 0, "token_starts second dim (n_pages) must be > 0");
  TORCH_CHECK(mid_tokens % n_pages_total == 0, "mid_tokens must be divisible by n_pages");
  const int64_t page_size = mid_tokens / n_pages_total;
  const int64_t page_end = page_begin + page_count;

  TORCH_CHECK(cpu_kv_linear.size(0) == bsz, "cpu_kv_linear batch must match gpu_mid_kv batch");
  TORCH_CHECK(cpu_kv_linear.size(2) == gpu_mid_kv.size(2), "kv dim mismatch");
  TORCH_CHECK(cpu_kv_linear.size(3) == gpu_mid_kv.size(3), "n_kv_heads mismatch");
  TORCH_CHECK(cpu_kv_linear.size(4) == gpu_mid_kv.size(4), "head_dim mismatch");
  TORCH_CHECK(starts_bsz == 1 || starts_bsz == bsz, "token_starts batch must be 1 or bsz");
  TORCH_CHECK(valid_tokens >= 0 && valid_tokens <= cpu_kv_linear.size(1), "valid_tokens out of range");

  const int64_t cpu_stride_tok = cpu_kv_linear.stride(1);
  const int64_t dst_stride_tok = gpu_mid_kv.stride(1);
  TORCH_CHECK(cpu_stride_tok == dst_stride_tok, "token stride mismatch between cpu_kv_linear and gpu_mid_kv");

  const int32_t* starts_ptr = starts_cpu.data_ptr<int32_t>();
  if (recall_bounds_check_enabled()) {
    for (int64_t sb = 0; sb < starts_bsz; ++sb) {
      for (int64_t p = page_begin; p < page_end; ++p) {
        const int64_t s = static_cast<int64_t>(starts_ptr[sb * n_pages_total + p]);
        TORCH_CHECK(
            s >= 0 && s + page_size <= valid_tokens,
            "token_starts out of valid range: start=",
            s,
            ", page_size=",
            page_size,
            ", valid_tokens=",
            valid_tokens);
      }
    }
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(gpu_mid_kv.scalar_type(), c_type, [&] {
    using scalar_t = c_type;
    scalar_t* dst_base = reinterpret_cast<scalar_t*>(gpu_mid_kv.data_ptr());
    scalar_t* src_base = reinterpret_cast<scalar_t*>(cpu_kv_linear.data_ptr());

    const size_t token_bytes = static_cast<size_t>(dst_stride_tok) * sizeof(scalar_t);
    const size_t page_bytes = static_cast<size_t>(page_size) * token_bytes;
    const int64_t src_stride_b = cpu_kv_linear.stride(0);
    const int64_t dst_stride_b = gpu_mid_kv.stride(0);

    for (int64_t b = 0; b < bsz; ++b) {
      const int64_t sb = (starts_bsz == 1) ? 0 : b;
      int64_t p = page_begin;
      while (p < page_end) {
        const int64_t p0 = p;
        const int64_t s0 = static_cast<int64_t>(starts_ptr[sb * n_pages_total + p0]);
        int64_t run_pages = 1;
        while (p0 + run_pages < page_end) {
          const int64_t sn = static_cast<int64_t>(starts_ptr[sb * n_pages_total + p0 + run_pages]);
          if (sn != s0 + run_pages * page_size) {
            break;
          }
          ++run_pages;
        }
        scalar_t* src_ptr = src_base + b * src_stride_b + s0 * cpu_stride_tok;
        scalar_t* dst_ptr = dst_base + b * dst_stride_b + (p0 * page_size) * dst_stride_tok;
        memcpy_async_checked(
            dst_ptr,
            src_ptr,
            page_bytes * static_cast<size_t>(run_pages),
            cudaMemcpyHostToDevice,
            stream,
            "recall_tokens_linear_partial");
        p += run_pages;
      }
    }
    return true;
  });
}

void recall_tokens_delta_linear_partial(
    const torch::Tensor &token_starts,
    const torch::Tensor &prev_token_starts,
    const torch::Tensor &cpu_kv_linear,
    const torch::Tensor &gpu_prev_mid_kv,
    torch::Tensor gpu_mid_kv,
    int64_t valid_tokens,
    int64_t page_begin,
    int64_t page_count
) {
  TORCH_CHECK(!cpu_kv_linear.is_cuda(), "cpu_kv_linear must be CPU tensor");
  CHECK_CONTIGUOUS(cpu_kv_linear);
  TORCH_CHECK(cpu_kv_linear.is_pinned(), "cpu_kv_linear must be pinned CPU tensor");
  TORCH_CHECK(cpu_kv_linear.dim() == 5, "cpu_kv_linear must be [bsz, max_tokens, 2, n_kv_heads, head_dim]");
  CHECK_INPUT(gpu_prev_mid_kv);
  CHECK_INPUT(gpu_mid_kv);
  TORCH_CHECK(gpu_prev_mid_kv.dim() == 5, "gpu_prev_mid_kv must be [bsz, mid_tokens, 2, n_kv_heads, head_dim]");
  TORCH_CHECK(gpu_mid_kv.dim() == 5, "gpu_mid_kv must be [bsz, mid_tokens, 2, n_kv_heads, head_dim]");
  TORCH_CHECK(gpu_prev_mid_kv.sizes() == gpu_mid_kv.sizes(), "gpu_prev_mid_kv and gpu_mid_kv must have identical shape");
  TORCH_CHECK(gpu_prev_mid_kv.scalar_type() == gpu_mid_kv.scalar_type(), "gpu_prev_mid_kv and gpu_mid_kv dtype mismatch");

  auto starts_cpu = to_cpu_i32_starts(token_starts, "token_starts");
  auto prev_starts_cpu = to_cpu_i32_starts(prev_token_starts, "prev_token_starts");
  const int64_t starts_bsz = starts_cpu.size(0);
  const int64_t prev_starts_bsz = prev_starts_cpu.size(0);
  const int64_t n_pages_total = starts_cpu.size(1);
  TORCH_CHECK(prev_starts_cpu.size(1) == n_pages_total, "token_starts and prev_token_starts must have same n_pages");

  const int64_t bsz = gpu_mid_kv.size(0);
  const int64_t mid_tokens = gpu_mid_kv.size(1);
  TORCH_CHECK(page_begin >= 0, "page_begin must be >= 0");
  TORCH_CHECK(page_count >= 0, "page_count must be >= 0");
  TORCH_CHECK(page_begin + page_count <= n_pages_total, "page_begin + page_count exceeds token_starts second dim");
  if (page_count == 0) {
    return;
  }
  TORCH_CHECK(n_pages_total > 0, "token_starts second dim (n_pages) must be > 0");
  TORCH_CHECK(mid_tokens % n_pages_total == 0, "mid_tokens must be divisible by n_pages");
  const int64_t page_size = mid_tokens / n_pages_total;
  const int64_t page_end = page_begin + page_count;

  TORCH_CHECK(cpu_kv_linear.size(0) == bsz, "cpu_kv_linear batch must match gpu_mid_kv batch");
  TORCH_CHECK(cpu_kv_linear.size(2) == gpu_mid_kv.size(2), "kv dim mismatch");
  TORCH_CHECK(cpu_kv_linear.size(3) == gpu_mid_kv.size(3), "n_kv_heads mismatch");
  TORCH_CHECK(cpu_kv_linear.size(4) == gpu_mid_kv.size(4), "head_dim mismatch");
  TORCH_CHECK(starts_bsz == 1 || starts_bsz == bsz, "token_starts batch must be 1 or bsz");
  TORCH_CHECK(prev_starts_bsz == 1 || prev_starts_bsz == bsz, "prev_token_starts batch must be 1 or bsz");
  TORCH_CHECK(valid_tokens >= 0 && valid_tokens <= cpu_kv_linear.size(1), "valid_tokens out of range");

  const int64_t cpu_stride_tok = cpu_kv_linear.stride(1);
  const int64_t prev_stride_tok = gpu_prev_mid_kv.stride(1);
  const int64_t dst_stride_tok = gpu_mid_kv.stride(1);
  TORCH_CHECK(cpu_stride_tok == dst_stride_tok, "token stride mismatch between cpu_kv_linear and gpu_mid_kv");
  TORCH_CHECK(prev_stride_tok == dst_stride_tok, "token stride mismatch between gpu_prev_mid_kv and gpu_mid_kv");

  const int32_t* starts_ptr = starts_cpu.data_ptr<int32_t>();
  const int32_t* prev_starts_ptr = prev_starts_cpu.data_ptr<int32_t>();
  if (recall_bounds_check_enabled()) {
    for (int64_t sb = 0; sb < starts_bsz; ++sb) {
      for (int64_t p = page_begin; p < page_end; ++p) {
        const int64_t s = static_cast<int64_t>(starts_ptr[sb * n_pages_total + p]);
        TORCH_CHECK(
            s >= 0 && s + page_size <= valid_tokens,
            "token_starts out of valid range: start=",
            s,
            ", page_size=",
            page_size,
            ", valid_tokens=",
            valid_tokens);
      }
    }
    for (int64_t sb = 0; sb < prev_starts_bsz; ++sb) {
      for (int64_t p = page_begin; p < page_end; ++p) {
        const int64_t s = static_cast<int64_t>(prev_starts_ptr[sb * n_pages_total + p]);
        TORCH_CHECK(
            s >= 0 && s + page_size <= valid_tokens,
            "prev_token_starts out of valid range: start=",
            s,
            ", page_size=",
            page_size,
            ", valid_tokens=",
            valid_tokens);
      }
    }
  }

  const bool same_buffer = gpu_prev_mid_kv.data_ptr() == gpu_mid_kv.data_ptr();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(gpu_mid_kv.scalar_type(), c_type, [&] {
    using scalar_t = c_type;
    scalar_t* dst_base = reinterpret_cast<scalar_t*>(gpu_mid_kv.data_ptr());
    scalar_t* prev_base = reinterpret_cast<scalar_t*>(gpu_prev_mid_kv.data_ptr());
    scalar_t* src_base = reinterpret_cast<scalar_t*>(cpu_kv_linear.data_ptr());

    const int64_t src_stride_b = cpu_kv_linear.stride(0);
    const int64_t prev_stride_b = gpu_prev_mid_kv.stride(0);
    const int64_t dst_stride_b = gpu_mid_kv.stride(0);
    const size_t token_bytes = static_cast<size_t>(dst_stride_tok) * sizeof(scalar_t);
    const size_t page_bytes = static_cast<size_t>(page_size) * token_bytes;
    const size_t page_pitch_bytes = static_cast<size_t>(page_size) * token_bytes;

    for (int64_t b = 0; b < bsz; ++b) {
      const int64_t sb_new = (starts_bsz == 1) ? 0 : b;
      const int64_t sb_old = (prev_starts_bsz == 1) ? 0 : b;

      if (!same_buffer) {
        scalar_t* dst_batch = dst_base + b * dst_stride_b;
        scalar_t* prev_batch = prev_base + b * prev_stride_b;
        memcpy_async_checked(
            dst_batch,
            prev_batch,
            page_bytes * static_cast<size_t>(n_pages_total),
            cudaMemcpyDeviceToDevice,
            stream,
            "recall_tokens_delta_linear_partial(d2d_full_prev)");

        int64_t p2 = page_begin;
        while (p2 < page_end) {
          const int64_t s_new = static_cast<int64_t>(starts_ptr[sb_new * n_pages_total + p2]);
          const int64_t s_old = static_cast<int64_t>(prev_starts_ptr[sb_old * n_pages_total + p2]);
          const int64_t shift = s_new - s_old;
          if (shift == 0) {
            ++p2;
            continue;
          }
          scalar_t* dst_page = dst_batch + (p2 * page_size) * dst_stride_tok;
          scalar_t* prev_page = prev_batch + (p2 * page_size) * prev_stride_tok;

          if (shift >= page_size || shift <= -page_size) {
            int64_t run_pages = 1;
            while (p2 + run_pages < page_end) {
              const int64_t sn = static_cast<int64_t>(starts_ptr[sb_new * n_pages_total + p2 + run_pages]);
              const int64_t so = static_cast<int64_t>(prev_starts_ptr[sb_old * n_pages_total + p2 + run_pages]);
              const int64_t sh = sn - so;
              if (sh != shift) break;
              if (sn != s_new + run_pages * page_size) break;
              if (so != s_old + run_pages * page_size) break;
              ++run_pages;
            }
            scalar_t* src_ptr = src_base + b * src_stride_b + s_new * cpu_stride_tok;
            memcpy_async_checked(
                dst_page,
                src_ptr,
                page_bytes * static_cast<size_t>(run_pages),
                cudaMemcpyHostToDevice,
                stream,
                "recall_tokens_delta_linear_partial(h2d_full)");
            p2 += run_pages;
            continue;
          }

          int64_t run_pages = 1;
          while (p2 + run_pages < page_end) {
            const int64_t sn = static_cast<int64_t>(starts_ptr[sb_new * n_pages_total + p2 + run_pages]);
            const int64_t so = static_cast<int64_t>(prev_starts_ptr[sb_old * n_pages_total + p2 + run_pages]);
            const int64_t sh = sn - so;
            if (sh != shift) break;
            if (sn != s_new + run_pages * page_size) break;
            if (so != s_old + run_pages * page_size) break;
            ++run_pages;
          }
          if (shift > 0) {
            const int64_t overlap = page_size - shift;
            scalar_t* d2d_src = prev_page + shift * prev_stride_tok;
            memcpy2d_async_checked(
                dst_page,
                page_pitch_bytes,
                d2d_src,
                page_pitch_bytes,
                static_cast<size_t>(overlap) * token_bytes,
                static_cast<size_t>(run_pages),
                cudaMemcpyDeviceToDevice,
                stream,
                "recall_tokens_delta_linear_partial(d2d_tail)");
            scalar_t* h2d_src = src_base + b * src_stride_b + (s_new + overlap) * cpu_stride_tok;
            scalar_t* h2d_dst = dst_page + overlap * dst_stride_tok;
            memcpy2d_async_checked(
                h2d_dst,
                page_pitch_bytes,
                h2d_src,
                page_pitch_bytes,
                static_cast<size_t>(shift) * token_bytes,
                static_cast<size_t>(run_pages),
                cudaMemcpyHostToDevice,
                stream,
                "recall_tokens_delta_linear_partial(h2d_tail)");
          } else {
            const int64_t shift_abs = -shift;
            const int64_t overlap = page_size - shift_abs;
            scalar_t* d2d_dst = dst_page + shift_abs * dst_stride_tok;
            memcpy2d_async_checked(
                d2d_dst,
                page_pitch_bytes,
                prev_page,
                page_pitch_bytes,
                static_cast<size_t>(overlap) * token_bytes,
                static_cast<size_t>(run_pages),
                cudaMemcpyDeviceToDevice,
                stream,
                "recall_tokens_delta_linear_partial(d2d_head)");
            scalar_t* h2d_src = src_base + b * src_stride_b + s_new * cpu_stride_tok;
            memcpy2d_async_checked(
                dst_page,
                page_pitch_bytes,
                h2d_src,
                page_pitch_bytes,
                static_cast<size_t>(shift_abs) * token_bytes,
                static_cast<size_t>(run_pages),
                cudaMemcpyHostToDevice,
                stream,
                "recall_tokens_delta_linear_partial(h2d_head)");
          }
          p2 += run_pages;
        }
      } else {
        int64_t p2 = page_begin;
        while (p2 < page_end) {
          const int64_t s_new = static_cast<int64_t>(starts_ptr[sb_new * n_pages_total + p2]);
          const int64_t s_old = static_cast<int64_t>(prev_starts_ptr[sb_old * n_pages_total + p2]);
          const int64_t shift = s_new - s_old;
          if (shift == 0) {
            ++p2;
            continue;
          }
          scalar_t* page_ptr = dst_base + b * dst_stride_b + (p2 * page_size) * dst_stride_tok;
          if (shift >= page_size || shift <= -page_size) {
            scalar_t* src_ptr = src_base + b * src_stride_b + s_new * cpu_stride_tok;
            memcpy_async_checked(
                page_ptr,
                src_ptr,
                page_bytes,
                cudaMemcpyHostToDevice,
                stream,
                "recall_tokens_delta_linear_partial(h2d_full_inplace)");
            ++p2;
            continue;
          }
          if (shift > 0) {
            const int64_t overlap = page_size - shift;
            scalar_t* d2d_src = page_ptr + shift * dst_stride_tok;
            cudaError_t err = cudaMemcpy(
                page_ptr, d2d_src, static_cast<size_t>(overlap) * token_bytes, cudaMemcpyDeviceToDevice);
            TORCH_CHECK(err == cudaSuccess, "recall_tokens_delta_linear_partial d2d inplace tail failed: ", cudaGetErrorString(err));
            scalar_t* h2d_src = src_base + b * src_stride_b + (s_new + overlap) * cpu_stride_tok;
            scalar_t* h2d_dst = page_ptr + overlap * dst_stride_tok;
            memcpy_async_checked(
                h2d_dst,
                h2d_src,
                static_cast<size_t>(shift) * token_bytes,
                cudaMemcpyHostToDevice,
                stream,
                "recall_tokens_delta_linear_partial(h2d_tail_inplace)");
          } else {
            const int64_t shift_abs = -shift;
            const int64_t overlap = page_size - shift_abs;
            cudaError_t err = cudaMemcpy(
                page_ptr + shift_abs * dst_stride_tok,
                page_ptr,
                static_cast<size_t>(overlap) * token_bytes,
                cudaMemcpyDeviceToDevice);
            TORCH_CHECK(err == cudaSuccess, "recall_tokens_delta_linear_partial d2d inplace head failed: ", cudaGetErrorString(err));
            scalar_t* h2d_src = src_base + b * src_stride_b + s_new * cpu_stride_tok;
            memcpy_async_checked(
                page_ptr,
                h2d_src,
                static_cast<size_t>(shift_abs) * token_bytes,
                cudaMemcpyHostToDevice,
                stream,
                "recall_tokens_delta_linear_partial(h2d_head_inplace)");
          }
          ++p2;
        }
      }
    }
    return true;
  });
}
