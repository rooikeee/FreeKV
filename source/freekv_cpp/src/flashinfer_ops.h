/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAEvent.h>

#include <flashinfer/attention/handler.cuh>
#include <flashinfer/layout.cuh>

torch::Tensor rms_norm(
	torch::Tensor input, // [bsz, len, hidden_dim]
	torch::Tensor weight, // [hidden_dim]
	float epsilon);

void qk_apply_rotary_in_place(
    torch::Tensor q, // [bsz, len, n_qo_heads, head_dim]
    torch::Tensor k, // [bsz, len, n_kv_heads, head_dim]
    unsigned int past_kv_len, 
    float rope_scale, 
    float rope_theta);

void qkq_apply_rotary_in_place(
    torch::Tensor q, // [bsz, len, n_qo_heads, head_dim]
    torch::Tensor k, // [bsz, len, n_kv_heads, head_dim]
    torch::Tensor q1, // [bsz, len, n_qo_heads, head_dim]
    unsigned int past_kv_len, 
    float rope_scale, 
    float rope_theta);

void append_paged_kv_cache_prefill(
	torch::Tensor k, // [bsz, kv_len, n_kv_heads, head_dim]
	torch::Tensor v, // [bsz, kv_len, n_kv_heads, head_dim]
	torch::Tensor kv_data, // [n_max_pages, 2, page_size, n_kv_heads, head_dim]
	torch::Tensor kv_indices, // [bsz, num_pages]
	torch::Tensor kv_indptr, // [bsz+1]
	torch::Tensor kv_last_page_len, // [bsz]
	unsigned int layout);

void append_paged_kv_cache_decode(
	torch::Tensor k, // [bsz, kv_len, n_kv_heads, head_dim]
	torch::Tensor v, // [bsz, kv_len, n_kv_heads, head_dim]
	torch::Tensor kv_data, // [n_max_pages, 2, page_size, n_kv_heads, head_dim]
	torch::Tensor kv_indices, // [bsz, num_pages]
	torch::Tensor kv_indptr, // [bsz+1]
	torch::Tensor kv_last_page_len, // [bsz]
	unsigned int layout);

torch::Tensor estimate_scores(
	torch::Tensor q, // [bsz, 1, num_heads, head_dim]
	torch::Tensor dg_data, // [n_max_pages, 2, page_size, n_kv_heads, head_dim]
	torch::Tensor dg_indices, // [bsz, num_pages]
	torch::Tensor dg_indptr, // [bsz+1]
	torch::Tensor dg_last_page_len, // [bsz]
	unsigned int dg_seq_len, 
	unsigned int layout,
  unsigned int n_groups);

void select_topk(
	torch::Tensor scores, // [bsz, len]
	torch::Tensor out_data, // [bsz, topk]
	torch::Tensor out_inds, // [bsz, topk + ns + nw]
	torch::Tensor new_in, // [bsz, cap]
	torch::Tensor incache, // [bsz, len]
	torch::Tensor pos_ids, // [bsz, cap]
	torch::Tensor recall_ids, // [bsz, topk + 1]
	torch::Tensor buf,
	unsigned int topk,
	unsigned int n_sink_pages,
	unsigned int n_win_pages);

void prefill_select_topk(
  torch::Tensor scores, // [bsz, len]
  torch::Tensor out_data, // [bsz, topk]
  torch::Tensor out_inds, // [bsz, topk + ns + nw = cap]
  torch::Tensor incache, // [bsz, len + 1]
  torch::Tensor incache1, // [bsz, len + 1]
  torch::Tensor pos_ids, // [bsz, cap]
  torch::Tensor buf,
  unsigned int topk,
  unsigned int n_sink_pages,
  unsigned int n_win_pages);

class BatchPrefillWithPagedKVCachePyTorchWrapper {
 public:
  void BeginForward(torch::Tensor workspace_buffer, torch::Tensor qo_indptr,
                    torch::Tensor page_kv_indptr, torch::Tensor page_kv_last_page_len,
                    unsigned int batch_size, unsigned int num_qo_heads, unsigned int num_kv_heads,
                    unsigned int head_dim, unsigned page_size, torch::Tensor empty_q_data);
  void EndForward();
  bool IsCUDAGraphEnabled() const { return handler_->IsCUDAGraphEnabled(); }
  std::vector<torch::Tensor> Forward(torch::Tensor q, torch::Tensor qo_indptr,
                                     torch::Tensor paged_kv_data, torch::Tensor paged_kv_indptr,
                                     torch::Tensor paged_kv_indices,
                                     torch::Tensor paged_kv_last_page_len, bool causal,
                                     unsigned int pos_encoding_mode, bool logits_cap,
                                     bool allow_fp16_qk_reduction, float sm_scale, float rope_scale,
                                     float rope_theta, bool return_lse);
  BatchPrefillWithPagedKVCachePyTorchWrapper(unsigned int layout, bool enable_cuda_graph)
      : kv_layout_(flashinfer::QKVLayout(layout)),
        handler_(std::make_shared<flashinfer::BatchPrefillHandler>(enable_cuda_graph)) {}

 private:
  std::shared_ptr<flashinfer::BatchPrefillHandler> handler_;
  flashinfer::QKVLayout kv_layout_;
};

class BatchDecodeWithPagedKVCachePyTorchWrapper {
 public:
  void BeginForward(torch::Tensor workspace_buffer, torch::Tensor indptr,
                    torch::Tensor last_page_len, unsigned int batch_size, unsigned int num_qo_heads,
                    unsigned int num_kv_heads, unsigned int head_dim, unsigned int page_size,
                    unsigned int pos_encoding_mode, bool logits_cap, torch::Tensor empty_q_data,
                    torch::Tensor empty_kv_data);
  void UpdatePageLockedBufferSize(uint32_t max_workspace_size_in_bytes);
  bool IsCUDAGraphEnabled() const { return handler_->IsCUDAGraphEnabled(); }
  void EndForward();
  std::vector<torch::Tensor> Forward(torch::Tensor q, torch::Tensor paged_kv_data,
                                     torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
                                     torch::Tensor paged_kv_last_page_len,
                                     unsigned int pos_encoding_mode, bool logits_cap,
                                     float sm_scale, float rope_scale, float rope_theta,
                                     bool return_lse);
  BatchDecodeWithPagedKVCachePyTorchWrapper(
      std::shared_ptr<flashinfer::BatchDecodeHandler> handler_ptr, flashinfer::QKVLayout kv_layout)
      : handler_(handler_ptr), kv_layout_(kv_layout) {}
  BatchDecodeWithPagedKVCachePyTorchWrapper(unsigned int layout, bool enable_cuda_graph,
                                            unsigned int fixed_batch_size)
      : kv_layout_(flashinfer::QKVLayout(layout)),
        handler_(std::make_shared<flashinfer::BatchDecodeHandler>(enable_cuda_graph,
                                                                  fixed_batch_size)) {}

 private:
  std::shared_ptr<flashinfer::BatchDecodeHandler> handler_;
  flashinfer::QKVLayout kv_layout_;
};

void recall_cuda_knl(
		const torch::Tensor &rids_gpu,
		const torch::Tensor &eids_gpu,
		const torch::Tensor &cpu_c2p,
		const torch::Tensor &cpu_buffer_pinned, // [max_pages, 2, page_size, n_kv_heads, head_dim]
		torch::Tensor kvc_buffer_gpu,						// [budget, 2, page_size, n_kv_heads, head_dim]
		int64_t n_groups,
		int64_t gs,
		int64_t nw,
		unsigned int cpu_layout,
    torch::Tensor recall_buf
);

void recall_torch_cpy(
		const torch::Tensor &rids_gpu,
		const torch::Tensor &eids_gpu,
		const torch::Tensor &cpu_c2p,
		const torch::Tensor &cpu_buffer_pinned, // [max_pages, 2, page_size, n_kv_heads, head_dim]
		torch::Tensor kvc_buffer_gpu,						// [budget, 2, page_size, n_kv_heads, head_dim]
		int64_t n_groups,
		int64_t gs,
		int64_t nw,
		unsigned int cpu_layout,
    torch::Tensor recall_buf
);

void recall_cuda_cpy(
		const torch::Tensor &rids_gpu,
		const torch::Tensor &eids_gpu,
		const torch::Tensor &cpu_c2p,
		const torch::Tensor &cpu_buffer_pinned, // [max_pages, 2, page_size, n_kv_heads, head_dim]
		torch::Tensor kvc_buffer_gpu,						// [budget, 2, page_size, n_kv_heads, head_dim]
		int64_t n_groups,
		int64_t gs,
		int64_t nw,
		unsigned int cpu_layout,
    torch::Tensor recall_buf
);

void recall_cuda_cpy_cpuhnd_2buf(
		const torch::Tensor &rids_gpu,
		const torch::Tensor &eids_gpu,
		const torch::Tensor &cpu_c2p,
		const torch::Tensor &cpu_buffer_pinned, // [max_pages, n_kv_heads, 2, page_size, head_dim]
		torch::Tensor kvc_buffer_gpu,						// [budget, 2, page_size, n_kv_heads, head_dim]
		int64_t n_groups,
		int64_t gs,
		int64_t nw,
		torch::Tensor recall_buf1,
		torch::Tensor recall_buf2,
    uint64_t stream_handle1,
    uint64_t stream_handle2,
		const torch::Tensor &need_recall_corr   // [bsz, n_kv_heads]
);

void recall_cuda_cpy_cpuhnd_2buf_core(
		const torch::Tensor &rids_gpu,
		const torch::Tensor &eids_gpu,
		const torch::Tensor &cpu_c2p,
		const torch::Tensor &cpu_buffer_pinned, // [max_pages, n_kv_heads, 2, page_size, head_dim]
		torch::Tensor kvc_buffer_gpu,						// [budget, 2, page_size, n_kv_heads, head_dim]
		int64_t n_groups,
		int64_t gs,
		int64_t nw,
		torch::Tensor recall_buf1,              // [budget, gs, 2, page_size, head_dim]
		torch::Tensor recall_buf2,
    cudaStream_t stream1,
    cudaStream_t stream2,
    cudaEvent_t event1,
    cudaEvent_t event2,
		const torch::Tensor &need_recall_corr = torch::Tensor()   // [bsz, n_kv_heads]
);

void recall_cuda_cpy_cpuhnd_2buf_pool(
		const torch::Tensor &rids_gpu,
		const torch::Tensor &eids_gpu,
		const torch::Tensor &cpu_c2p,
		const torch::Tensor &cpu_buffer_pinned, // [max_pages, n_kv_heads, 2, page_size, head_dim]
		torch::Tensor kvc_buffer_gpu,						// [budget, 2, page_size, n_kv_heads, head_dim]
		int64_t n_groups,
		int64_t gs,
		int64_t nw,
		torch::Tensor recall_buf1,              // [budget, gs, 2, page_size, head_dim]
		torch::Tensor recall_buf2,
    uint64_t stream_handle1,
    uint64_t stream_handle2,
    uint64_t event_handle1,
    uint64_t event_handle2,
		const torch::Tensor &need_recall_corr   // [bsz, n_kv_heads]
);
void init_recall_thread_pool(int num_threads);
void shutdown_recall_thread_pool();

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
);

torch::Tensor alloc_managed_bool(int64_t rows, int64_t cols);

torch::Tensor alloc_managed_bool_scalar();

bool get_corr_managed_cuda(
    const torch::Tensor &query_states,
    const torch::Tensor &last_step_q,
    int64_t n_kv_heads,
    float corr,
    torch::Tensor to_corr_managed
);
