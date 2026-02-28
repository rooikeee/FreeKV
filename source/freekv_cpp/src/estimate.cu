#include <torch/extension.h>
#include "estimate.cuh"

torch::Tensor estimate_scores(torch::Tensor q, // [bsz, 1, num_heads, head_dim]
                              torch::Tensor dg_data,
                              torch::Tensor dg_indices, // [bsz, num_pages]
                              torch::Tensor dg_indptr,  // [bsz+1]
                              torch::Tensor dg_last_page_len, // [bsz]
                              unsigned int dg_seq_len, unsigned int layout,
                              unsigned int n_groups) {
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
  int32_t page_size, num_kv_heads;

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  if (kv_layout == QKVLayout::kHND) {
    page_size = dg_data.size(3);
    num_kv_heads = dg_data.size(2);
    CHECK_EQ(dg_data.size(4), head_dim);
  } else {
    page_size = dg_data.size(2);
    num_kv_heads = dg_data.size(3);
    CHECK_EQ(dg_data.size(4), head_dim);
  }
  cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor o =
      torch::empty({batch_size, num_qo_heads, dg_seq_len}, q.options());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    return DISPATCH_kv_layout(kv_layout, KV_LAYOUT, [&] {
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

  // if (n_groups == 1)
  //   return o.mean(1);
  CHECK_EQ(num_qo_heads % n_groups, 0);
  return o.reshape({batch_size, n_groups, num_qo_heads / n_groups, dg_seq_len})
      .mean(2);
}
