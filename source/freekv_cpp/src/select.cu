#include "select.cuh"

void select_topk(torch::Tensor scores,     // [bsz, len]
                 torch::Tensor out_data,   // [bsz, topk]
                 torch::Tensor out_inds,   // [bsz, topk + ns + nw]
                 torch::Tensor new_in,     // [bsz, cap]
                 torch::Tensor incache,    // [bsz, len + 1]
                 torch::Tensor pos_ids,    // [bsz, cap]
                 torch::Tensor recall_ids, // [bsz, topk + 1]
                 torch::Tensor buf, unsigned int topk,
                 unsigned int n_sink_pages, unsigned int n_win_pages) {

  CHECK_INPUT(scores);
  CHECK_INPUT(out_data);
  CHECK_INPUT(out_inds);
  CHECK_INPUT(new_in);
  CHECK_INPUT(incache);
  CHECK_INPUT(pos_ids);
  CHECK_INPUT(recall_ids);
  CHECK_INPUT(buf);

  CHECK_DIM(2, scores);
  CHECK_DIM(2, out_data);
  CHECK_DIM(2, out_inds);
  CHECK_DIM(2, new_in);
  CHECK_DIM(2, incache);
  CHECK_DIM(2, pos_ids);
  CHECK_DIM(2, recall_ids);

  int batch_size = scores.size(0);
  int len = scores.size(1);
  int cap = pos_ids.size(1);

  CHECK_GE(len, topk);
  CHECK_EQ(topk, out_data.size(1));
  CHECK_EQ(topk + n_sink_pages + n_win_pages, out_inds.size(1));
  CHECK_EQ(topk + 1, recall_ids.size(1));
  CHECK_EQ(cap, new_in.size(1));
  CHECK_EQ(len + 1, incache.size(1));
  CHECK_EQ(batch_size, out_data.size(0));
  CHECK_EQ(batch_size, out_inds.size(0));
  CHECK_EQ(batch_size, new_in.size(0));
  CHECK_EQ(batch_size, incache.size(0));
  CHECK_EQ(batch_size, pos_ids.size(0));
  CHECK_EQ(batch_size, recall_ids.size(0));

  CHECK_EQ(out_inds.scalar_type(), torch::kInt32);
  CHECK_EQ(incache.scalar_type(), torch::kInt32);
  CHECK_EQ(pos_ids.scalar_type(), torch::kInt32);
  CHECK_EQ(recall_ids.scalar_type(), torch::kInt32);

  cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(scores.scalar_type(), c_type, [&] {
    SelectTopk<c_type, int32_t>(static_cast<c_type *>(scores.data_ptr()),
                                static_cast<c_type *>(out_data.data_ptr()),
                                static_cast<int32_t *>(out_inds.data_ptr()),
                                static_cast<c_type *>(new_in.data_ptr()),
                                static_cast<int32_t *>(incache.data_ptr()),
                                static_cast<int32_t *>(pos_ids.data_ptr()),
                                static_cast<int32_t *>(recall_ids.data_ptr()),
                                static_cast<char *>(buf.data_ptr()), batch_size,
                                len, cap, topk, n_sink_pages, n_win_pages,
                                /*select_min=*/false, curr_stream);
    return true;
  });
}

void prefill_select_topk(torch::Tensor scores,   // [bsz, len]
                         torch::Tensor out_data, // [bsz, topk]
                         torch::Tensor out_inds, // [bsz, topk + ns + nw = cap]
                         torch::Tensor incache,  // [bsz, len + 1]
                         torch::Tensor incache1, // [bsz, len + 1]
                         torch::Tensor pos_ids,  // [bsz, cap]
                         torch::Tensor buf, unsigned int topk,
                         unsigned int n_sink_pages, unsigned int n_win_pages) {

  CHECK_INPUT(scores);
  CHECK_INPUT(out_data);
  CHECK_INPUT(out_inds);
  CHECK_INPUT(incache);
  CHECK_INPUT(pos_ids);

  CHECK_DIM(2, scores);
  CHECK_DIM(2, out_data);
  CHECK_DIM(2, out_inds);
  CHECK_DIM(2, incache);
  CHECK_DIM(2, pos_ids);

  int batch_size = scores.size(0);
  int len = scores.size(1);
  int cap = pos_ids.size(1);

  CHECK_GE(len, topk);
  CHECK_EQ(topk, out_data.size(1));
  CHECK_EQ(topk + n_sink_pages + n_win_pages, out_inds.size(1));
  CHECK_EQ(cap, topk + n_sink_pages + n_win_pages);
  CHECK_EQ(len + 1, incache.size(1));
  CHECK_EQ(batch_size, out_data.size(0));
  CHECK_EQ(batch_size, out_inds.size(0));
  CHECK_EQ(batch_size, incache.size(0));
  CHECK_EQ(batch_size, pos_ids.size(0));

  CHECK_EQ(out_inds.scalar_type(), torch::kInt32);
  CHECK_EQ(incache.scalar_type(), torch::kInt32);
  CHECK_EQ(pos_ids.scalar_type(), torch::kInt32);

  cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(scores.scalar_type(), c_type, [&] {
    PrefillSelectTopk<c_type, int32_t>(
        static_cast<c_type *>(scores.data_ptr()),
        static_cast<c_type *>(out_data.data_ptr()),
        static_cast<int32_t *>(out_inds.data_ptr()),
        static_cast<int32_t *>(incache.data_ptr()),
        static_cast<int32_t *>(incache1.data_ptr()),
        static_cast<int32_t *>(pos_ids.data_ptr()),
        static_cast<char *>(buf.data_ptr()), batch_size, len, cap, topk,
        n_sink_pages, n_win_pages, /*select_min=*/false, curr_stream);
    return true;
  });
}
