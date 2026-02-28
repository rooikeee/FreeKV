#include <torch/extension.h>
#include <cuda_runtime.h>
#include <optional>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/Dispatch.h>
#include "pytorch_extension_utils.h"

#include <flashinfer/attention/decode.cuh>
#include "recall_kernels.cuh"

enum RecallImplementation {
    CUDA_KERNEL,
    TORCH_CPY,
    CUDA_CPY_ASYNC
};


template <RecallImplementation impl, QKVLayout cpu_layout>
void recall_impl(
		// Input Tensors
		const torch::Tensor &rids_gpu,
		const torch::Tensor &eids_gpu,
		const torch::Tensor &cpu_c2p,
		const torch::Tensor &cpu_buffer_pinned, // [max_pages, 2, page_size, n_kv_heads, head_dim]
                                            // or [max_pages, n_kv_heads, 2, page_size, head_dim]
		torch::Tensor kvc_buffer_gpu,						// [budget, 2, page_size, n_kv_heads, head_dim]
		int64_t n_groups,
		int64_t gs,
		int64_t nw,
		torch::Tensor recall_buf
)
{
	CHECK_INPUT(kvc_buffer_gpu);
	CHECK_CONTIGUOUS(cpu_buffer_pinned);

	int64_t batch_size = rids_gpu.size(0);

	void *cpu_buffer_mapped_ptr_void = nullptr;
  if constexpr (impl == RecallImplementation::CUDA_KERNEL) {
    cudaError_t err = cudaHostGetDevicePointer(&cpu_buffer_mapped_ptr_void,
                                              const_cast<void *>(cpu_buffer_pinned.data_ptr()),
                                              0);
    TORCH_CHECK(err == cudaSuccess, 
      "cudaHostGetDevicePointer failed! Ensure CPU buffer is pinned and driver/runtime supports mapping. Error: ", 
      cudaGetErrorString(err));
  }

	cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

	const int64_t page_size = kvc_buffer_gpu.size(2);
	const int64_t n_kv_heads = kvc_buffer_gpu.size(3);
	const int64_t head_dim = kvc_buffer_gpu.size(-1);
	const int64_t stride_page = kvc_buffer_gpu.stride(0);
	const int64_t stride_kv = kvc_buffer_gpu.stride(1);
	const int64_t stride_heads = kvc_buffer_gpu.stride(2);
  TORCH_CHECK(cpu_buffer_pinned.stride(0) == stride_page);
  const int64_t cpu_kvc_stride1 = cpu_buffer_pinned.stride(1);
  const int64_t cpu_kvc_stride2 = cpu_buffer_pinned.stride(2);

	// Accessors for CPU index tensors
	auto cpu_c2p_acc = cpu_c2p.accessor<int32_t, 2>();
  auto rids_cpu = rids_gpu.to(torch::kCPU);
	auto rids_acc = rids_cpu.accessor<int32_t, 3>();
  // not need for CUDA_CPY_ASYNC and HND CPU
  torch::Tensor eids_cpu;
	std::optional<torch::TensorAccessor<int32_t, 3>> eids_acc;
  if constexpr (impl == RecallImplementation::CUDA_CPY_ASYNC && cpu_layout == QKVLayout::kHND) {
    CHECK_INPUT(recall_buf);
    TORCH_CHECK(recall_buf.dim() == 5);
    const int64_t stride_rbuf_page = recall_buf.stride(0);
    TORCH_CHECK(stride_rbuf_page == gs*2*page_size*head_dim);
  }
  else {
    eids_cpu = eids_gpu.to(torch::kCPU);
    eids_acc.emplace(eids_cpu.accessor<int32_t, 3>());
  }

	for (int64_t i = 0; i < batch_size; ++i) {
		for (int64_t j = 0; j < n_groups; ++j) {
			const int32_t nr = rids_acc[i][j][0];
			if (nr <= 0) {
				continue;
			}

			int64_t head_start = j * gs;
			int64_t head_end = (j + 1) * gs;

			int64_t r_indices_start = 1;
			int64_t r_indices_end = 1 + nr;
			int64_t eids_len_total = eids_gpu.size(2);
			int64_t e_indices_start = eids_len_total - (nr + nw);
			int64_t e_indices_end = eids_len_total - nw;

			// Basic Bounds checks
			TORCH_CHECK(r_indices_start >= 0 && r_indices_end <= rids_gpu.size(2), "rids slice out of bounds for i=%ld, j=%ld", i, j);
			TORCH_CHECK(e_indices_start >= 0 && e_indices_end <= eids_gpu.size(2) && e_indices_start <= e_indices_end, "eids slice out of bounds for i=%ld, j=%ld", i, j);

			for (int64_t k = 0; k < nr; ++k) {
				int64_t ei;
        if constexpr (!(impl == RecallImplementation::CUDA_CPY_ASYNC && cpu_layout == QKVLayout::kHND)) {
          ei = (*eids_acc)[i][j][e_indices_start + k];
        }

				// Translate ri to source pool index (rpi_cpu)
				int64_t ri = rids_acc[i][j][r_indices_start + k];
				int64_t rpi_cpu;
				rpi_cpu = static_cast<int64_t>(cpu_c2p_acc[i][ri]);

				if constexpr (impl == RecallImplementation::TORCH_CPY) {
          auto dest_page_view = kvc_buffer_gpu.select(0, ei);
          auto dest_slice_view = dest_page_view.slice(2, head_start, head_end);
          auto source_page_view = cpu_buffer_pinned.select(0, rpi_cpu);
          if constexpr (cpu_layout == QKVLayout::kNHD) {
            auto source_slice_cpu = source_page_view.slice(2, head_start, head_end);
            dest_slice_view.copy_(source_slice_cpu, /*non_blocking=*/true);
          }
          else {  // cpu_layout == QKVLayout::kHND
            auto source_slice_cpu = source_page_view.slice(0, head_start, head_end);
            if (gs == 1) {
              // [gs, 2, page_size, head_dim] => [2, page_size, gs, head_dim]
              auto source_slice_cpu_nhd = source_slice_cpu.view({2, page_size, gs, head_dim});
              dest_slice_view.copy_(source_slice_cpu_nhd, /*non_blocking=*/true);
            }
            else {
              TORCH_CHECK(false, "not needed yet")
            }
          } 
				}
				else if constexpr (impl == RecallImplementation::CUDA_KERNEL) {
					const int total_elements = 2 * page_size * gs * head_dim;
					const int threads_per_block = 128;
					const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

					AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
						kvc_buffer_gpu.scalar_type(), "recall_kernel", [&] {
						// Pass the mapped CPU pointer cast to the correct type
						scalar_t* cpu_buffer_mapped_ptr = static_cast<scalar_t*>(cpu_buffer_mapped_ptr_void);
						copy_single_page_slice_kernel<scalar_t, cpu_layout>
              <<<blocks_per_grid, threads_per_block, 0, curr_stream>>>(
								kvc_buffer_gpu.data_ptr<scalar_t>(),
								cpu_buffer_mapped_ptr,
								ei,
								rpi_cpu,
								head_start, head_end, head_dim, page_size,
								stride_page, stride_kv, stride_heads,
                cpu_kvc_stride1, cpu_kvc_stride2,
								total_elements
						); 
					});
					cudaError_t loop_err = cudaGetLastError();
					TORCH_CHECK(loop_err == cudaSuccess, 
						"CUDA kernel launch failed in inner loop (i=", i, ", j=", j, ", k=", k, "): ", 
						cudaGetErrorString(loop_err));
				}
				else if constexpr (impl == RecallImplementation::CUDA_CPY_ASYNC) {
					AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
						kvc_buffer_gpu.scalar_type(), "recall_memcpy", [&] {
            scalar_t* base_cpu_ptr = 
              const_cast<scalar_t*>(cpu_buffer_pinned.data_ptr<scalar_t>()) + rpi_cpu * stride_page;
					  const	size_t elem_size = sizeof(scalar_t);

            if constexpr (cpu_layout == QKVLayout::kNHD) {
						  scalar_t* base_gpu_ptr = kvc_buffer_gpu.data_ptr<scalar_t>() + ei * stride_page;
              size_t copy_width_bytes = head_dim * elem_size;
						  size_t copy_height = gs;
              // Pitch = stride between start of one row (head) and start of next row (head) IN BYTES
              size_t pitch_bytes = head_dim * elem_size;

              for (int kv_idx = 0; kv_idx < 2; ++kv_idx) {
                for (int p_idx = 0; p_idx < page_size; ++p_idx) {
                  // Calculate starting pointer for this specific 2D slice (start of head_start)
                  // Pointer to [ei/rpi, kv, p, head_start, 0]
                  size_t offset = kv_idx * stride_kv + p_idx * stride_heads + head_start * head_dim;
                  scalar_t* dst_ptr = base_gpu_ptr + offset;
                  scalar_t* src_ptr = base_cpu_ptr + offset;

                  cudaError_t copy_err = cudaMemcpy2DAsync(
                      dst_ptr,
                      pitch_bytes,
                      src_ptr,
                      pitch_bytes,
                      copy_width_bytes,
                      copy_height,
                      cudaMemcpyHostToDevice,
                      curr_stream
                  );
                  // Check error immediately after each call for debugging
                  TORCH_CHECK(copy_err == cudaSuccess, "cudaMemcpy2DAsync failed (i=", i, ", j=", j, ", k=", k, ", kv=", kv_idx, ", p=", p_idx, "): ", cudaGetErrorString(copy_err));
                }
              }
            }
            else {
              const int64_t stride_rbuf_page = recall_buf.stride(0);
              cudaMemcpyAsync(recall_buf.data_ptr<scalar_t>() + k * stride_rbuf_page,
                              base_cpu_ptr + head_start * cpu_kvc_stride1,
                              stride_rbuf_page * elem_size, cudaMemcpyHostToDevice, curr_stream);
            }
					});
				}
				else {
					TORCH_CHECK(false, "not allowed recall implementation");
				}
			}
      if constexpr (impl == RecallImplementation::CUDA_CPY_ASYNC && cpu_layout == QKVLayout::kHND) {
        if (gs == 1) {
          cudaError_t err = cudaStreamSynchronize(curr_stream);
          TORCH_CHECK(err == cudaSuccess, 
            "streamSync failed in inner loop (i=", i, ", j=", j,
            cudaGetErrorString(err));
          const auto eids_gpu_ij = eids_gpu[i][j].slice(0, e_indices_start, e_indices_end);
          CHECK_INPUT(eids_gpu_ij);

          const int total_elements = nr * 2 * page_size * head_dim;
          const int threads_per_block = 256;
          const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
          AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            recall_buf.scalar_type(), "dispatch", [&] {
            dispatch_recall_buf_kernel<scalar_t>
              <<<blocks_per_grid, threads_per_block, 0, curr_stream>>>(
                const_cast<scalar_t*>(recall_buf.data_ptr<scalar_t>()),
                kvc_buffer_gpu.data_ptr<scalar_t>(),
                const_cast<int32_t*>(eids_gpu_ij.data_ptr<int32_t>()),
                recall_buf.stride(0),
                stride_page,
                page_size,
                n_kv_heads,
                head_dim,
                head_start,
                nr,
                total_elements
            );
          });
          cudaError_t loop_err = cudaGetLastError();
          TORCH_CHECK(loop_err == cudaSuccess, 
            "dispatch CUDA kernel launch failed in inner loop (i=", i, ", j=", j, 
            cudaGetErrorString(loop_err));
        }
        else {
          TORCH_CHECK(false, "not needed yet")
        }
      }
		}
	}
	cudaError_t err = cudaStreamSynchronize(curr_stream);
  TORCH_CHECK(err == cudaSuccess, "final stream sync fail", cudaGetErrorString(err));
}

void recall_cuda_cpy_cpuhnd_1buf(
		const torch::Tensor &rids_gpu,
		const torch::Tensor &eids_gpu,
		const torch::Tensor &cpu_c2p,
		const torch::Tensor &cpu_buffer_pinned, // [max_pages, n_kv_heads, 2, page_size, head_dim]
		torch::Tensor kvc_buffer_gpu,						// [budget, 2, page_size, n_kv_heads, head_dim]
		int64_t n_groups,
		int64_t gs,
		int64_t nw,
		torch::Tensor recall_buf
) {
	CHECK_INPUT(kvc_buffer_gpu);
	CHECK_CONTIGUOUS(cpu_buffer_pinned);
  CHECK_INPUT(recall_buf);
  TORCH_CHECK(recall_buf.dim() == 5);

	int64_t batch_size = rids_gpu.size(0);

	cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();
	const int64_t page_size = kvc_buffer_gpu.size(2);
	const int64_t n_kv_heads = kvc_buffer_gpu.size(3);
	const int64_t head_dim = kvc_buffer_gpu.size(4);
	const int64_t stride_page = kvc_buffer_gpu.stride(0);
  TORCH_CHECK(stride_page == cpu_buffer_pinned.stride(0));
  const int64_t cpu_kvc_stride1 = cpu_buffer_pinned.stride(1);

  const int64_t stride_rbuf_page = recall_buf.stride(0);
  TORCH_CHECK(stride_rbuf_page == gs*2*page_size*head_dim);

	auto cpu_c2p_acc = cpu_c2p.accessor<int32_t, 2>();
  auto rids_cpu = rids_gpu.to(torch::kCPU);
	auto rids_acc = rids_cpu.accessor<int32_t, 3>();

	const int64_t r_indices_start = 1;
	const int64_t eids_len_total = eids_gpu.size(2);
	const int64_t e_indices_end = eids_len_total - nw;

	for (int64_t i = 0; i < batch_size; ++i) {
		for (int64_t j = 0; j < n_groups; ++j) {
			const int32_t nr = rids_acc[i][j][0];
			if (nr <= 0) {
				continue;
			}

			int64_t head_start = j * gs;
			const int64_t e_indices_start = e_indices_end - nr;

			for (int64_t k = 0; k < nr; ++k) {
				int64_t ri = rids_acc[i][j][r_indices_start + k];
				int64_t rpi_cpu = static_cast<int64_t>(cpu_c2p_acc[i][ri]);
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
          kvc_buffer_gpu.scalar_type(), "recall_memcpy", [&] {

          const size_t elem_size = sizeof(scalar_t);
          scalar_t* base_cpu_ptr = 
            const_cast<scalar_t*>(cpu_buffer_pinned.data_ptr<scalar_t>()) + rpi_cpu * stride_page;
          cudaMemcpyAsync(recall_buf.data_ptr<scalar_t>() + k * stride_rbuf_page,
                          base_cpu_ptr + head_start * cpu_kvc_stride1,
                          stride_rbuf_page * elem_size, cudaMemcpyHostToDevice, curr_stream);
        });
      }
      cudaError_t err = cudaStreamSynchronize(curr_stream);
      TORCH_CHECK(err == cudaSuccess, 
        "streamSync failed in inner loop (i=", i, ", j=", j,
        cudaGetErrorString(err));
      if (gs == 1) {
        const auto eids_gpu_ij = eids_gpu[i][j].slice(0, e_indices_start, e_indices_end);
        CHECK_INPUT(eids_gpu_ij);

        const int total_elements = nr * 2 * page_size * head_dim;
        const int threads_per_block = 256;
        const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
          recall_buf.scalar_type(), "dispatch", [&] {

          dispatch_recall_buf_kernel<scalar_t>
            <<<blocks_per_grid, threads_per_block, 0, curr_stream>>>(
              const_cast<scalar_t*>(recall_buf.data_ptr<scalar_t>()),
              kvc_buffer_gpu.data_ptr<scalar_t>(),
              const_cast<int32_t*>(eids_gpu_ij.data_ptr<int32_t>()),
              stride_rbuf_page,
              stride_page,
              page_size,
              n_kv_heads,
              head_dim,
              head_start,
              nr,
              total_elements
          );
        });
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, 
          "dispatch CUDA kernel launch failed in inner loop (i=", i, ", j=", j, 
          cudaGetErrorString(err));
      }
      else {
        TORCH_CHECK(false, "not needed yet")
      }
		}
	}
  cudaError_t err = cudaStreamSynchronize(curr_stream);
  TORCH_CHECK(err == cudaSuccess, "final stream sync fail", cudaGetErrorString(err));
}

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
		torch::Tensor recall_buf						// empty or [budget, gs, 2, page_size, head_dim]
) {
  QKVLayout cpu_kv_layout = static_cast<QKVLayout>(cpu_layout);
  if (cpu_kv_layout == QKVLayout::kNHD)
    recall_impl<RecallImplementation::CUDA_KERNEL, QKVLayout::kNHD>(
      rids_gpu, eids_gpu, cpu_c2p, cpu_buffer_pinned,
      kvc_buffer_gpu, n_groups, gs, nw, recall_buf
    );
  else
    recall_impl<RecallImplementation::CUDA_KERNEL, QKVLayout::kHND>(
      rids_gpu, eids_gpu, cpu_c2p, cpu_buffer_pinned,
      kvc_buffer_gpu, n_groups, gs, nw, recall_buf
    );
}

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
		torch::Tensor recall_buf						// empty or [budget, gs, 2, page_size, head_dim]
) {
  QKVLayout cpu_kv_layout = static_cast<QKVLayout>(cpu_layout);
  if (cpu_kv_layout == QKVLayout::kNHD)
    recall_impl<RecallImplementation::TORCH_CPY, QKVLayout::kNHD>(
      rids_gpu, eids_gpu, cpu_c2p, cpu_buffer_pinned,
      kvc_buffer_gpu, n_groups, gs, nw, recall_buf
    );
  else
    recall_impl<RecallImplementation::TORCH_CPY, QKVLayout::kHND>(
      rids_gpu, eids_gpu, cpu_c2p, cpu_buffer_pinned,
      kvc_buffer_gpu, n_groups, gs, nw, recall_buf
    );
}

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
		torch::Tensor recall_buf						// empty or [budget, gs, 2, page_size, head_dim]
) {
  QKVLayout cpu_kv_layout = static_cast<QKVLayout>(cpu_layout);
  if (cpu_kv_layout == QKVLayout::kNHD)
    recall_impl<RecallImplementation::CUDA_CPY_ASYNC, QKVLayout::kNHD>(
      rids_gpu, eids_gpu, cpu_c2p, cpu_buffer_pinned,
      kvc_buffer_gpu, n_groups, gs, nw, recall_buf
    );
  else
    recall_cuda_cpy_cpuhnd_1buf(
      rids_gpu, eids_gpu, cpu_c2p, cpu_buffer_pinned,
      kvc_buffer_gpu, n_groups, gs, nw, recall_buf
    );
}
