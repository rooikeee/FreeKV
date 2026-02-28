#include <flashinfer/attention/prefill.cuh>

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

void BatchPrefillWithPagedKVCachePyTorchWrapper::BeginForward(
    torch::Tensor workspace_buffer, torch::Tensor qo_indptr, torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_last_page_len, unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int num_kv_heads, unsigned int head_dim, unsigned int page_size,
    torch::Tensor empty_q_data) {
  // NOTE(Zihao): not necessary to be a CUDA tensor
  CHECK_CONTIGUOUS(qo_indptr);
  CHECK_CONTIGUOUS(workspace_buffer);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  CHECK_DIM(1, qo_indptr);
  CHECK_DIM(1, workspace_buffer);

  qo_indptr = qo_indptr.to(torch::kInt32);
  paged_kv_indptr = paged_kv_indptr.to(torch::kInt32);
  paged_kv_last_page_len = paged_kv_last_page_len.to(torch::kInt32);
  size_t workspace_size_in_bytes = workspace_buffer.size(0) * workspace_buffer.element_size();
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  handler_->SetCUDAStream(torch_current_stream);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(empty_q_data.scalar_type(), q_type, [&] {
    cudaError_t status = handler_->BeginForward<q_type, int32_t>(
        static_cast<void*>(workspace_buffer.data_ptr()), workspace_size_in_bytes,
        static_cast<int32_t*>(qo_indptr.data_ptr()),
        static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
        static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()), batch_size, num_qo_heads,
        num_kv_heads, head_dim, page_size);
    TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithPagedKVCache failed with error ",
                cudaGetErrorString(status));
    return true;
  });
}

void BatchPrefillWithPagedKVCachePyTorchWrapper::EndForward() {
  handler_->EndForward();
}

template <PageStorage PAGE_STORAGE, uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK,
          QKVLayout KV_LAYOUT, PosEncodingMode POS_ENCODING_MODE, bool ALLOW_FP16_QK_REDUCTION,
          MaskMode MASK_MODE, typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* q_indptr, IdType* q_offset,
    paged_kv_t<PAGE_STORAGE, KV_LAYOUT, DTypeIn, IdType> paged_kv, uint8_t* custom_mask,
    IdType* qk_indptr, DTypeOut* o, float* lse, uint32_t num_qo_heads, float sm_scale,
    float rope_scale, float rope_theta, cudaStream_t stream) {
  DTypeOut* tmp_v = nullptr;
  float* tmp_s = nullptr;
  IdType *request_indices = nullptr, *qo_tile_indices = nullptr, *kv_tile_indices = nullptr,
         *o_indptr = nullptr, *merge_indptr = nullptr, *kv_chunk_size_ptr = nullptr;
  bool* block_valid_mask = nullptr;
  uint32_t num_frags_x = 0U;
  uint32_t padded_batch_size = 0U;
  uint32_t total_num_rows = 0U;
  if (handler->IsForwardStarted()) {
    tmp_v = handler->GetTempV<DTypeOut>();
    tmp_s = handler->GetTempS();
    request_indices = handler->GetRequestIndices<IdType>();
    qo_tile_indices = handler->GetQOTileIndices<IdType>();
    kv_tile_indices = handler->GetKVTileIndices<IdType>();
    block_valid_mask = handler->GetBlockValidMask();
    o_indptr = handler->GetOIndptr<IdType>();
    merge_indptr = handler->GetMergeIndptr<IdType>();
    kv_chunk_size_ptr = handler->GetKVChunkSizePtr<IdType>();
    num_frags_x = handler->GetNumFragsX();
    padded_batch_size = handler->GetPaddedBatchSize();
    total_num_rows = handler->GetTotalNumRows();
  } else {
    std::ostringstream err_msg;
    err_msg << "Please call BatchPrefillHandler's BeginForward() before calling "
               "BatchPrefillWithPagedKVCacheWrapper()";
    throw std::runtime_error(err_msg.str());
  }

  DISPATCH_NUM_FRAGS_X(num_frags_x, NUM_FRAGS_X, {
    return BatchPrefillWithPagedKVCacheDispatched<
        PAGE_STORAGE, NUM_FRAGS_X, HEAD_DIM, LOGITS_POST_HOOK, KV_LAYOUT, POS_ENCODING_MODE,
        ALLOW_FP16_QK_REDUCTION, MASK_MODE, DTypeIn, DTypeOut, IdType>(
        q, request_indices, qo_tile_indices, kv_tile_indices, q_indptr, q_offset, paged_kv,
        custom_mask, qk_indptr, o_indptr, o, tmp_v, tmp_s, lse, merge_indptr, block_valid_mask,
        kv_chunk_size_ptr, total_num_rows, num_qo_heads, padded_batch_size, sm_scale, rope_scale,
        rope_theta, stream);
  });
  return cudaSuccess;
}

std::vector<torch::Tensor> BatchPrefillWithPagedKVCachePyTorchWrapper::Forward(
    torch::Tensor q, torch::Tensor qo_indptr, torch::Tensor paged_kv_data,
    torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len, bool causal, unsigned int pos_encoding_mode,
    bool logits_cap, bool allow_fp16_qk_reduction, float sm_scale, float rope_scale,
    float rope_theta, bool return_lse) {
  CHECK_INPUT(q);
  CHECK_INPUT(qo_indptr);
  CHECK_INPUT(paged_kv_data);
  CHECK_INPUT(paged_kv_indptr);
  CHECK_INPUT(paged_kv_indices);
  CHECK_INPUT(paged_kv_last_page_len);
  CHECK_DIM(3, q);         // (nnz_qo, H_qo, D)
  CHECK_DIM(1, qo_indptr); // (B + 1,)
  // [max_num_pages, 2, num_kv_heads, page_size, head_dim] for HND
  // [max_num_pages, 2, page_size, num_kv_heads, head_dim] for HND
  CHECK_DIM(5, paged_kv_data);
  CHECK_DIM(1, paged_kv_indptr);        // (B + 1,)
  CHECK_DIM(1, paged_kv_indices);       // (nnz_kv,)
  CHECK_DIM(1, paged_kv_last_page_len); // (B,)
  int64_t batch_size = qo_indptr.size(0) - 1;
  int64_t nnz_qo = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t num_kv_heads, page_size;
  if (kv_layout_ == QKVLayout::kHND) {
    num_kv_heads = paged_kv_data.size(2);
    page_size = paged_kv_data.size(3);
  } else {
    page_size = paged_kv_data.size(2);
    num_kv_heads = paged_kv_data.size(3);
  }
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  CHECK_EQ(qo_indptr.size(0), batch_size + 1);
  CHECK_EQ(paged_kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(paged_kv_last_page_len.size(0), batch_size);
  CHECK_EQ(paged_kv_data.size(1), 2);
  CHECK_EQ(paged_kv_data.size(4), head_dim);
  qo_indptr = qo_indptr.to(torch::kInt32);
  paged_kv_indptr = paged_kv_indptr.to(torch::kInt32);
  paged_kv_indices = paged_kv_indices.to(torch::kInt32);
  paged_kv_last_page_len = paged_kv_last_page_len.to(torch::kInt32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  torch::Tensor o = torch::empty_like(q, q.options());
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({nnz_qo, num_qo_heads}, q.options()).to(torch::kFloat32);
  }
  MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  const LogitsPostHook logits_post_hook =
      logits_cap ? LogitsPostHook::kCap30 : LogitsPostHook::kNone;

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
      return DISPATCH_kv_layout(kv_layout_, KV_LAYOUT, [&] {
        paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
            num_kv_heads, page_size, head_dim, batch_size,
            static_cast<c_type*>(paged_kv_data.data_ptr()),
            static_cast<int32_t*>(paged_kv_indices.data_ptr()),
            static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
            static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()));
        return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
          return DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {
            return DISPATCH_allow_fp16_qk_reduction(
                allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, [&] {
                  return DISPATCH_pos_encoding_mode(
                      PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                        cudaError_t status = BatchPrefillWithPagedKVCacheWrapperDispatched<
                            PageStorage::kIndices, HEAD_DIM, LOGITS_POST_HOOK, KV_LAYOUT,
                            POS_ENCODING_MODE, ALLOW_FP16_QK_REDUCTION, MASK_MODE, c_type, c_type,
                            int32_t>(
                            handler_.get(), static_cast<c_type*>(q.data_ptr()),
                            static_cast<int32_t*>(qo_indptr.data_ptr()),
                            /*q_offset=*/nullptr, paged_kv,
                            /*custom_mask=*/nullptr,
                            /*qk_indptr=*/nullptr, static_cast<c_type*>(o.data_ptr()),
                            /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
                            num_qo_heads, sm_scale, rope_scale, rope_theta,
                            /*stream=*/torch_current_stream);
                        TORCH_CHECK(status == cudaSuccess,
                                    "BatchPrefillWithPagedKVCache failed with error code ",
                                    cudaGetErrorString(status));
                        return true;
                      });
                });
          });
        });
      });
    });
  });

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}
