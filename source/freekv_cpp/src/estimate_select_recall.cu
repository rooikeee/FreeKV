#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/Dispatch.h>
#include "pytorch_extension_utils.h"

#include "estimate.cuh"
#include "select.cuh"
#include "flashinfer_ops.h"
#include "thread_pool.h"

void estimate_select_recall_core(
    // for estimate
    const torch::Tensor &q, // [bsz, 1, num_heads, head_dim]
    const torch::Tensor &dg_data, // [n_max_pages, 2, page_size, n_kv_heads, head_dim]
    const torch::Tensor &dg_indices, // [bsz, num_pages]
	  unsigned int dg_seq_len, 
    const torch::Tensor &dg_indptr, // [bsz+1]
    const torch::Tensor &dg_last_page_len, // [bsz]
    // for topk
    torch::Tensor out_data, // [sel_bsz, topk]
    torch::Tensor eids, // [sel_bsz, topk + ns + nw]
    torch::Tensor rids, // [sel_bsz, topk + 1]
    torch::Tensor new_in, // [sel_bsz, budget]
    torch::Tensor buf,
    const torch::Tensor &incache, // [sel_bsz, dg_seq_len + 1], cc2gp
    const torch::Tensor &pos_ids, // [sel_bsz, budget], gc2cc
    unsigned int topk,
    // for recall
		const torch::Tensor &cpu_c2p,
		const torch::Tensor &cpu_buffer_pinned, // [max_pages, n_kv_heads, 2, page_size, head_dim]
		torch::Tensor kvc_buffer_gpu,						// [budget, 2, page_size, n_kv_heads, head_dim]
    int64_t n_groups, 
    int64_t gs, 
    unsigned int n_sink_pages,
    unsigned int n_win_pages,
		torch::Tensor recall_buf1,              // [budget, gs, 2, page_size, head_dim]
		torch::Tensor recall_buf2,
    uint64_t stream_handle1,
    uint64_t stream_handle2,
    uint64_t event_handle1,
    uint64_t event_handle2
) {
  CHECK_INPUT(q);
  CHECK_INPUT(dg_data);
  CHECK_INPUT(dg_indices);
  CHECK_INPUT(dg_indptr);

  CHECK_DIM(4, q);
  CHECK_DIM(5, dg_data);
  CHECK_DIM(2, dg_indices);
  CHECK_DIM(1, dg_indptr);

  CHECK_EQ(q.size(1), 1);
  CHECK_EQ(dg_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(dg_indptr.scalar_type(), torch::kInt32);

  int32_t batch_size = q.size(0);
  int32_t num_qo_heads = q.size(2);
  int32_t head_dim = q.size(3);
  int32_t page_size = dg_data.size(2);
  int32_t num_kv_heads = dg_data.size(3);
  int cap = pos_ids.size(1);
  CHECK_EQ(num_kv_heads, n_groups);
  CHECK_EQ(dg_data.size(4), head_dim);

  int32_t sel_bsz = batch_size * n_groups;
  CHECK_INPUT(out_data);
  CHECK_INPUT(eids);
  CHECK_INPUT(new_in);
  CHECK_INPUT(incache);
  CHECK_INPUT(pos_ids);
  CHECK_INPUT(rids);
  CHECK_DIM(2, out_data);
  CHECK_DIM(2, eids);
  CHECK_DIM(2, new_in);
  CHECK_DIM(2, incache);
  CHECK_DIM(2, pos_ids);
  CHECK_DIM(2, rids);
  CHECK_GE(dg_seq_len, topk);
  CHECK_EQ(topk, out_data.size(1));
  CHECK_EQ(topk + n_sink_pages + n_win_pages, eids.size(1));
  CHECK_EQ(topk + 1, rids.size(1));
  CHECK_EQ(cap, new_in.size(1));
  CHECK_EQ(dg_seq_len + 1, incache.size(1));
  CHECK_EQ(sel_bsz, out_data.size(0));
  CHECK_EQ(sel_bsz, eids.size(0));
  CHECK_EQ(sel_bsz, new_in.size(0));
  CHECK_EQ(sel_bsz, incache.size(0));
  CHECK_EQ(sel_bsz, pos_ids.size(0));
  CHECK_EQ(sel_bsz, rids.size(0));
  CHECK_EQ(eids.scalar_type(), torch::kInt32);
  CHECK_EQ(incache.scalar_type(), torch::kInt32);
  CHECK_EQ(pos_ids.scalar_type(), torch::kInt32);
  CHECK_EQ(rids.scalar_type(), torch::kInt32);

  // put estimate and topk on recall stream 1, instead of compute stream
  cudaStream_t curr_stream = reinterpret_cast<cudaStream_t>(stream_handle1);
  cudaStream_t stream2 = reinterpret_cast<cudaStream_t>(stream_handle2);
  at::cuda::CUDAStreamGuard stream_guard(
    at::cuda::getStreamFromExternal(curr_stream, q.device().index())
  );

  torch::Tensor o =
    torch::empty({batch_size, num_qo_heads, dg_seq_len}, q.options());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    return DISPATCH_kv_layout(QKVLayout::kNHD, KV_LAYOUT, [&] {
      paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
          num_kv_heads, page_size, head_dim, batch_size,
          static_cast<c_type *>(dg_data.data_ptr()),
          static_cast<int32_t *>(dg_indices.data_ptr()),
          static_cast<int32_t *>(dg_indptr.data_ptr()),
          static_cast<int32_t *>(dg_last_page_len.data_ptr()));
      cudaError_t status =
          EstimateScores<PageStorage::kIndices, KV_LAYOUT, c_type, c_type,
                         int32_t>(static_cast<c_type *>(q.data_ptr()), paged_kv,
                                  static_cast<c_type *>(o.data_ptr()),
                                  num_qo_heads, PosEncodingMode::kNone, curr_stream);
      TORCH_CHECK(status == cudaSuccess,
                  "estimate_scores failed with error code ",
                  cudaGetErrorString(status));
      return true;
    });
  });

  torch::Tensor scores = 
    o.reshape({batch_size, n_groups, num_qo_heads / n_groups, dg_seq_len}).mean(2);
  scores = scores.reshape({sel_bsz, -1});
  CHECK_INPUT(scores);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(scores.scalar_type(), c_type, [&] {
    SelectTopk<c_type, int32_t>(static_cast<c_type *>(scores.data_ptr()),
                                static_cast<c_type *>(out_data.data_ptr()),
                                static_cast<int32_t *>(eids.data_ptr()),
                                static_cast<c_type *>(new_in.data_ptr()),
                                static_cast<int32_t *>(incache.data_ptr()),
                                static_cast<int32_t *>(pos_ids.data_ptr()),
                                static_cast<int32_t *>(rids.data_ptr()),
                                static_cast<char *>(buf.data_ptr()), sel_bsz,
                                dg_seq_len, cap, topk, n_sink_pages, n_win_pages,
                                /*select_min=*/false, curr_stream);
    return true;
  });

  rids = rids.reshape({batch_size, n_groups, -1});
  eids = eids.reshape({batch_size, n_groups, -1});
  TORCH_CHECK(rids.size(-1) == topk + 1);
  TORCH_CHECK(eids.size(-1) == topk + n_sink_pages + n_win_pages);

  cudaEvent_t event1 = reinterpret_cast<cudaEvent_t>(event_handle1);
  cudaEvent_t event2 = reinterpret_cast<cudaEvent_t>(event_handle2);
  
  recall_cuda_cpy_cpuhnd_2buf_core(
    rids, eids, cpu_c2p, cpu_buffer_pinned, kvc_buffer_gpu,
    n_groups, gs, n_win_pages,
    recall_buf1, recall_buf2,
    curr_stream, stream2, event1, event2
  );
}

void estimate_select_recall_pool(
    // for estimate
    const torch::Tensor &q, // [bsz, 1, num_heads, head_dim]
    const torch::Tensor &dg_data, // [n_max_pages, 2, page_size, n_kv_heads, head_dim]
    const torch::Tensor &dg_indices, // [bsz, num_pages]
	  unsigned int dg_seq_len, 
    const torch::Tensor &dg_indptr, // [bsz+1]
    const torch::Tensor &dg_last_page_len, // [bsz]
    // for topk
    torch::Tensor out_data, // [sel_bsz, topk]
    torch::Tensor eids, // [sel_bsz, topk + ns + nw]
    torch::Tensor rids, // [sel_bsz, topk + 1]
    torch::Tensor new_in, // [sel_bsz, budget]
    torch::Tensor buf,
    const torch::Tensor &incache, // [sel_bsz, dg_seq_len + 1], cc2gp
    const torch::Tensor &pos_ids, // [sel_bsz, budget], gc2cc
    unsigned int topk,
    // for recall
		const torch::Tensor &cpu_c2p,
		const torch::Tensor &cpu_buffer_pinned, // [max_pages, n_kv_heads, 2, page_size, head_dim]
		torch::Tensor kvc_buffer_gpu,						// [budget, 2, page_size, n_kv_heads, head_dim]
    int64_t n_groups, 
    int64_t gs, 
    unsigned int n_sink_pages,
    unsigned int n_win_pages,
		torch::Tensor recall_buf1,              // [budget, gs, 2, page_size, head_dim]
		torch::Tensor recall_buf2,
    uint64_t stream_handle1,
    uint64_t stream_handle2,
    uint64_t event_handle1,
    uint64_t event_handle2
) {
  TORCH_CHECK(recall_thread_pool);
  recall_thread_pool->enqueue(
    estimate_select_recall_core,
    q, dg_data, dg_indices, dg_seq_len, dg_indptr, dg_last_page_len,
    out_data, eids, rids, new_in, buf, incache, pos_ids, topk,
    cpu_c2p, cpu_buffer_pinned, kvc_buffer_gpu,
    n_groups, gs, n_sink_pages, n_win_pages,
    recall_buf1, recall_buf2, 
    stream_handle1, stream_handle2,
    event_handle1, event_handle2
  );
}
