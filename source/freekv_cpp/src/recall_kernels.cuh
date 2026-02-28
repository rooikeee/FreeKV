#pragma once

#include <cuda_runtime.h>


template <typename scalar_t, QKVLayout cpu_layout>
__global__ void copy_single_page_slice_kernel(
		scalar_t *__restrict__ dest_buffer,
		const scalar_t *__restrict__ source_buffer,
		int64_t dest_page_idx,
		int64_t src_page_idx,
		int head_start,
		int head_end,
		int head_dim,
		int page_size,
		int64_t stride_page,
		int64_t stride_kv,
		int64_t stride_heads,
		int64_t cpu_kvc_stride1,
		int64_t cpu_kvc_stride2,
		int total_elements
)
{
	// Calculate elements for this single page slice copy
	const int n_heads = head_end - head_start;
	// Linear thread index within this slice copy operation
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < total_elements)
	{
		const int d_idx = idx % head_dim;
		const int h_idx = (idx / head_dim) % n_heads;
		const int p_idx = (idx / head_dim / n_heads) % page_size;
		const int kv_idx = (idx / head_dim / n_heads / page_size) % 2;

		// Calculate offsets using the *single* page indices provided
		int64_t dest_offset = dest_page_idx * stride_page +
													kv_idx * stride_kv + 
													p_idx * stride_heads +
													(head_start + h_idx) * head_dim + d_idx;

    int64_t src_offset;
    if constexpr (cpu_layout == QKVLayout::kNHD) {
      src_offset = src_page_idx * stride_page +
                            kv_idx * stride_kv + 
                            p_idx * stride_heads +
                            (head_start + h_idx) * head_dim + d_idx;
    }
    else {
      src_offset = src_page_idx * stride_page +
                            (head_start + h_idx) * cpu_kvc_stride1 +
                            kv_idx * cpu_kvc_stride2 + 
                            p_idx * head_dim + d_idx;
    }

		dest_buffer[dest_offset] = source_buffer[src_offset];
	}
}


template <typename scalar_t>
__global__ void dispatch_recall_buf_kernel(
    const scalar_t *__restrict__ recall_buf,
    scalar_t *__restrict__ gpu_kvc_buf,
    const int *__restrict__ eids_gpu,
    int64_t stride_rbuf_page,
    int64_t stride_page,
    int64_t page_size,
    int64_t n_kv_heads,
    int64_t head_dim,
    int head_idx,
    int nr,
    int total_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    const int hd_idx = idx % head_dim;
    const int tk_idx = (idx / head_dim) % page_size;
    const int kv_idx = (idx / head_dim / page_size) % 2;
    const int rp_idx = idx / head_dim / page_size / 2;
    const int eid = eids_gpu[rp_idx];

    const int64_t src_offset = rp_idx * stride_rbuf_page +
                               kv_idx * page_size * head_dim +
                               tk_idx * head_dim + hd_idx;
    const int64_t dst_offset = eid * stride_page +
                               kv_idx * page_size * n_kv_heads * head_dim +
                               tk_idx * n_kv_heads * head_dim +
                               head_idx * head_dim + hd_idx;
    gpu_kvc_buf[dst_offset] = recall_buf[src_offset];
  }
}
