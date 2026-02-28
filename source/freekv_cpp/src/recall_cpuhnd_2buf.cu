#include <torch/extension.h>
#include <cuda_runtime.h>
#include <optional>
#include <vector>
#include <ATen/Dispatch.h>

#include "pytorch_extension_utils.h"
#include "thread_pool.h"
#include "recall_kernels.cuh"

void recall_cuda_cpy_cpuhnd_2buf(
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
		const torch::Tensor &need_recall_corr   // [bsz, n_kv_heads]
) {
  TORCH_CHECK(gs == 1);
	CHECK_INPUT(kvc_buffer_gpu);
	CHECK_CONTIGUOUS(cpu_buffer_pinned);
  CHECK_INPUT(recall_buf1);
  TORCH_CHECK(recall_buf1.dim() == 5);
  CHECK_INPUT(recall_buf2);
  TORCH_CHECK(recall_buf2.dim() == 5);

	int64_t batch_size = rids_gpu.size(0);

	const int64_t page_size = kvc_buffer_gpu.size(2);
	const int64_t n_kv_heads = kvc_buffer_gpu.size(3);
	const int64_t head_dim = kvc_buffer_gpu.size(4);
	const int64_t stride_page = kvc_buffer_gpu.stride(0);
  TORCH_CHECK(stride_page == cpu_buffer_pinned.stride(0));
  const int64_t cpu_kvc_stride1 = cpu_buffer_pinned.stride(1);

  const int64_t stride_rbuf_page = recall_buf1.stride(0);
  TORCH_CHECK(stride_rbuf_page == gs*2*page_size*head_dim);

	auto cpu_c2p_acc = cpu_c2p.accessor<int32_t, 2>();
  auto rids_cpu = rids_gpu.to(torch::kCPU);
	auto rids_acc = rids_cpu.accessor<int32_t, 3>();
  std::optional<torch::TensorAccessor<bool, 2>> need_recall_corr_acc_opt;
  if (need_recall_corr.defined() && need_recall_corr.dim() > 1) {
    need_recall_corr_acc_opt.emplace(need_recall_corr.accessor<bool, 2>());
  }

  cudaStream_t stream1 = reinterpret_cast<cudaStream_t>(stream_handle1);
  cudaStream_t stream2 = reinterpret_cast<cudaStream_t>(stream_handle2);
  TORCH_CHECK(stream1 != stream2);
  std::vector<torch::Tensor> recall_bufs = {recall_buf1, recall_buf2};
  std::vector<cudaStream_t> streams = {stream1, stream2};

	const int64_t r_indices_start = 1;
	const int64_t e_indices_end = eids_gpu.size(2) - nw;

  torch::Tensor prev_eids_gpu_slice;
  int64_t prev_nr = 0;
  int64_t prev_head_start = 0;
  int64_t prev_i = -1, prev_j = -1; // For error messages

	for (int64_t i = 0; i < batch_size; ++i) {
		for (int64_t j = 0; j < n_groups; ++j) {
      int curr_idx = (i*n_groups + j) % 2;
      int prev_idx = 1 - curr_idx;
      torch::Tensor curr_recall_buf = recall_bufs[curr_idx];
      cudaStream_t curr_stream = streams[curr_idx];

			const int32_t nr = rids_acc[i][j][0];
      // blocking version for correction, default all need correction
      bool cr = true;
      if (need_recall_corr_acc_opt.has_value()) {
        cr = (*need_recall_corr_acc_opt)[i][j];
      }
			if (nr <= 0 || !cr) { // not need correct
        // no cpy needed for current, maybe dispatch for prev
        if (prev_nr > 0) {
          torch::Tensor& dispatch_buf = recall_bufs[prev_idx];
          cudaStream_t dispatch_stream = streams[prev_idx];
          const int total_elements = prev_nr * 2 * page_size * head_dim;
          const int threads_per_block = 256;
          const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
          AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            dispatch_buf.scalar_type(), "dispatch_prev", [&] {
            dispatch_recall_buf_kernel<scalar_t>
              <<<blocks_per_grid, threads_per_block, 0, dispatch_stream>>>(
                const_cast<scalar_t*>(dispatch_buf.data_ptr<scalar_t>()),
                kvc_buffer_gpu.data_ptr<scalar_t>(),
                const_cast<int32_t*>(prev_eids_gpu_slice.data_ptr<int32_t>()),
                stride_rbuf_page, stride_page, page_size, n_kv_heads,
                head_dim, prev_head_start, prev_nr, total_elements);
          });
          prev_nr = 0;
        }
				continue;
			}

			int64_t head_start = j * gs;
			const int64_t e_indices_start = e_indices_end - nr;
      auto current_eids_gpu_slice = eids_gpu.index({
        i, j, torch::indexing::Slice(e_indices_start, e_indices_end)
      });
      CHECK_CONTIGUOUS(current_eids_gpu_slice);

      if (prev_nr > 0) {
        torch::Tensor& dispatch_buf = recall_bufs[prev_idx];
        cudaStream_t dispatch_stream = streams[prev_idx];
        const int total_elements = prev_nr * 2 * page_size * head_dim;
        const int threads_per_block = 256;
        const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
          dispatch_buf.scalar_type(), "dispatch_prev", [&] {
          dispatch_recall_buf_kernel<scalar_t>
            <<<blocks_per_grid, threads_per_block, 0, dispatch_stream>>>(
              const_cast<scalar_t*>(dispatch_buf.data_ptr<scalar_t>()),
              kvc_buffer_gpu.data_ptr<scalar_t>(),
              const_cast<int32_t*>(prev_eids_gpu_slice.data_ptr<int32_t>()),
              stride_rbuf_page, stride_page, page_size, n_kv_heads,
              head_dim, prev_head_start, prev_nr, total_elements);
        });
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, 
          "dispatch (prev) launch failed (i=", prev_i, ", j=", prev_j, "): ", cudaGetErrorString(err));
      }

			for (int64_t k = 0; k < nr; ++k) {
				int64_t ri = rids_acc[i][j][r_indices_start + k]; // Source cache index
				int64_t rpi_cpu = static_cast<int64_t>(cpu_c2p_acc[i][ri]);
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
          kvc_buffer_gpu.scalar_type(), "recall_memcpy", [&] {
          const size_t elem_size = sizeof(scalar_t);
          scalar_t* base_cpu_ptr = 
            const_cast<scalar_t*>(cpu_buffer_pinned.data_ptr<scalar_t>()) + rpi_cpu * stride_page;
          cudaMemcpyAsync(curr_recall_buf.data_ptr<scalar_t>() + k * stride_rbuf_page,
                          base_cpu_ptr + head_start * cpu_kvc_stride1,
                          stride_rbuf_page * elem_size, cudaMemcpyHostToDevice, curr_stream);
        });
      } // end loop k

      prev_eids_gpu_slice = std::move(current_eids_gpu_slice);
      prev_nr = nr;
      prev_head_start = head_start;
      prev_i = i;
      prev_j = j;
		} // end loop j
	} // end loop i

  // --- Dispatch for the VERY LAST group ---
  if (prev_nr > 0) {
    int prev_idx = (batch_size*n_groups - 1) % 2;
    torch::Tensor& dispatch_buf = recall_bufs[prev_idx];
    cudaStream_t dispatch_stream = streams[prev_idx];
    
    const int total_elements = prev_nr * 2 * page_size * head_dim;
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      dispatch_buf.scalar_type(), "dispatch_last", [&] {
      dispatch_recall_buf_kernel<scalar_t>
          <<<blocks_per_grid, threads_per_block, 0, dispatch_stream>>>(
            const_cast<scalar_t*>(dispatch_buf.data_ptr<scalar_t>()),
            kvc_buffer_gpu.data_ptr<scalar_t>(),
            const_cast<int32_t*>(prev_eids_gpu_slice.data_ptr<int32_t>()),
            stride_rbuf_page, stride_page, page_size, n_kv_heads,
            head_dim, prev_head_start, prev_nr, total_elements);
    });
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, 
      "dispatch (last) launch failed (i=", prev_i, ", j=", prev_j, "): ", cudaGetErrorString(err));
  }

  cudaError_t err1 = cudaStreamSynchronize(streams[0]);
  cudaError_t err2 = cudaStreamSynchronize(streams[1]);
  TORCH_CHECK(err1 == cudaSuccess && err2 == cudaSuccess,
              "Final stream sync failed. Stream1: ", cudaGetErrorString(err1),
              " Stream2: ", cudaGetErrorString(err2));
}

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
		const torch::Tensor &need_recall_corr = torch::Tensor() // [bsz, n_kv_heads]
) {
  TORCH_CHECK(gs == 1);
	CHECK_INPUT(kvc_buffer_gpu);
	CHECK_CONTIGUOUS(cpu_buffer_pinned);
  CHECK_INPUT(recall_buf1);
  TORCH_CHECK(recall_buf1.dim() == 5);
  CHECK_INPUT(recall_buf2);
  TORCH_CHECK(recall_buf2.dim() == 5);

	int64_t batch_size = rids_gpu.size(0);

	const int64_t page_size = kvc_buffer_gpu.size(2);
	const int64_t n_kv_heads = kvc_buffer_gpu.size(3);
	const int64_t head_dim = kvc_buffer_gpu.size(4);
	const int64_t stride_page = kvc_buffer_gpu.stride(0);
  TORCH_CHECK(stride_page == cpu_buffer_pinned.stride(0));
  const int64_t cpu_kvc_stride1 = cpu_buffer_pinned.stride(1);

  const int64_t stride_rbuf_page = recall_buf1.stride(0);
  TORCH_CHECK(stride_rbuf_page == gs*2*page_size*head_dim);

	auto cpu_c2p_acc = cpu_c2p.accessor<int32_t, 2>();
  auto rids_cpu = rids_gpu.to(torch::kCPU);
	auto rids_acc = rids_cpu.accessor<int32_t, 3>();
  std::optional<torch::TensorAccessor<bool, 2>> need_recall_corr_acc_opt;
  if (need_recall_corr.defined() && need_recall_corr.dim() > 1) {
    need_recall_corr_acc_opt.emplace(need_recall_corr.accessor<bool, 2>());
  }

  TORCH_CHECK(stream1 != stream2);
  std::vector<torch::Tensor> recall_bufs = {recall_buf1, recall_buf2};
  std::vector<cudaStream_t> streams = {stream1, stream2};

	const int64_t r_indices_start = 1;
	const int64_t e_indices_end = eids_gpu.size(2) - nw;

  torch::Tensor prev_eids_gpu_slice;
  int64_t prev_nr = 0;
  int64_t prev_head_start = 0;
  int64_t prev_i = -1, prev_j = -1; // For error messages

	for (int64_t i = 0; i < batch_size; ++i) {
		for (int64_t j = 0; j < n_groups; ++j) {
      int curr_idx = (i*n_groups + j) % 2;
      int prev_idx = 1 - curr_idx;
      torch::Tensor curr_recall_buf = recall_bufs[curr_idx];
      cudaStream_t curr_stream = streams[curr_idx];

			const int32_t nr = rids_acc[i][j][0];
      // non-blocking version for overlap, default: all are not corrected
      bool cr = false;
      if (need_recall_corr_acc_opt.has_value()) {
        cr = (*need_recall_corr_acc_opt)[i][j];
      }
			if (nr <= 0 || cr) {  // corrected, not need recall anymore
        // no cpy needed for current, maybe dispatch for prev
        if (prev_nr > 0) {
          torch::Tensor& dispatch_buf = recall_bufs[prev_idx];
          cudaStream_t dispatch_stream = streams[prev_idx];
          const int total_elements = prev_nr * 2 * page_size * head_dim;
          const int threads_per_block = 256;
          const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
          AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            dispatch_buf.scalar_type(), "dispatch_prev", [&] {
            dispatch_recall_buf_kernel<scalar_t>
              <<<blocks_per_grid, threads_per_block, 0, dispatch_stream>>>(
                const_cast<scalar_t*>(dispatch_buf.data_ptr<scalar_t>()),
                kvc_buffer_gpu.data_ptr<scalar_t>(),
                const_cast<int32_t*>(prev_eids_gpu_slice.data_ptr<int32_t>()),
                stride_rbuf_page, stride_page, page_size, n_kv_heads,
                head_dim, prev_head_start, prev_nr, total_elements);
          });
          prev_nr = 0;
        }
				continue;
			}

			int64_t head_start = j * gs;
			const int64_t e_indices_start = e_indices_end - nr;
      auto current_eids_gpu_slice = eids_gpu.index({
        i, j, torch::indexing::Slice(e_indices_start, e_indices_end)
      });
      CHECK_CONTIGUOUS(current_eids_gpu_slice);

      if (prev_nr > 0) {
        torch::Tensor& dispatch_buf = recall_bufs[prev_idx];
        cudaStream_t dispatch_stream = streams[prev_idx];
        const int total_elements = prev_nr * 2 * page_size * head_dim;
        const int threads_per_block = 256;
        const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
          dispatch_buf.scalar_type(), "dispatch_prev", [&] {
          dispatch_recall_buf_kernel<scalar_t>
            <<<blocks_per_grid, threads_per_block, 0, dispatch_stream>>>(
              const_cast<scalar_t*>(dispatch_buf.data_ptr<scalar_t>()),
              kvc_buffer_gpu.data_ptr<scalar_t>(),
              const_cast<int32_t*>(prev_eids_gpu_slice.data_ptr<int32_t>()),
              stride_rbuf_page, stride_page, page_size, n_kv_heads,
              head_dim, prev_head_start, prev_nr, total_elements);
        });
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, 
          "dispatch (prev) launch failed (i=", prev_i, ", j=", prev_j, "): ", cudaGetErrorString(err));
      }

			for (int64_t k = 0; k < nr; ++k) {
				int64_t ri = rids_acc[i][j][r_indices_start + k]; // Source cache index
				int64_t rpi_cpu = static_cast<int64_t>(cpu_c2p_acc[i][ri]);
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
          kvc_buffer_gpu.scalar_type(), "recall_memcpy", [&] {
          const size_t elem_size = sizeof(scalar_t);
          scalar_t* base_cpu_ptr = 
            const_cast<scalar_t*>(cpu_buffer_pinned.data_ptr<scalar_t>()) + rpi_cpu * stride_page;
          cudaMemcpyAsync(curr_recall_buf.data_ptr<scalar_t>() + k * stride_rbuf_page,
                          base_cpu_ptr + head_start * cpu_kvc_stride1,
                          stride_rbuf_page * elem_size, cudaMemcpyHostToDevice, curr_stream);
        });
      } // end loop k

      prev_eids_gpu_slice = std::move(current_eids_gpu_slice);
      prev_nr = nr;
      prev_head_start = head_start;
      prev_i = i;
      prev_j = j;
		} // end loop j
	} // end loop i

  // --- Dispatch for the VERY LAST group ---
  if (prev_nr > 0) {
    int prev_idx = (batch_size*n_groups - 1) % 2;
    torch::Tensor& dispatch_buf = recall_bufs[prev_idx];
    cudaStream_t dispatch_stream = streams[prev_idx];
    
    const int total_elements = prev_nr * 2 * page_size * head_dim;
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      dispatch_buf.scalar_type(), "dispatch_last", [&] {
      dispatch_recall_buf_kernel<scalar_t>
          <<<blocks_per_grid, threads_per_block, 0, dispatch_stream>>>(
            const_cast<scalar_t*>(dispatch_buf.data_ptr<scalar_t>()),
            kvc_buffer_gpu.data_ptr<scalar_t>(),
            const_cast<int32_t*>(prev_eids_gpu_slice.data_ptr<int32_t>()),
            stride_rbuf_page, stride_page, page_size, n_kv_heads,
            head_dim, prev_head_start, prev_nr, total_elements);
    });
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, 
      "dispatch (last) launch failed (i=", prev_i, ", j=", prev_j, "): ", cudaGetErrorString(err));
  }

  cudaError_t err1 = cudaEventRecord(event1, stream1);
  cudaError_t err2 = cudaEventRecord(event2, stream2);

  TORCH_CHECK(err1 == cudaSuccess && err2 == cudaSuccess,
              "Final event record failed. Stream1: ", cudaGetErrorString(err1),
              " Stream2: ", cudaGetErrorString(err2));
}

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
) {
  TORCH_CHECK(recall_thread_pool);
  cudaStream_t stream1 = reinterpret_cast<cudaStream_t>(stream_handle1);
  cudaStream_t stream2 = reinterpret_cast<cudaStream_t>(stream_handle2);
  cudaEvent_t event1 = reinterpret_cast<cudaEvent_t>(event_handle1);
  cudaEvent_t event2 = reinterpret_cast<cudaEvent_t>(event_handle2);
  recall_thread_pool->enqueue(
    recall_cuda_cpy_cpuhnd_2buf_core,
    rids_gpu, eids_gpu, cpu_c2p, cpu_buffer_pinned, kvc_buffer_gpu,
    n_groups, gs, nw, recall_buf1, recall_buf2,
    stream1, stream2, event1, event2,
    need_recall_corr
  );
}
