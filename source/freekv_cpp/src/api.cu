#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include "flashinfer_ops.h"

PYBIND11_MODULE(freekv_cpp, m) {
	m.def("rms_norm", &rms_norm);
	m.def("qk_apply_rotary_in_place", &qk_apply_rotary_in_place);
	m.def("qkq_apply_rotary_in_place", &qkq_apply_rotary_in_place);
	m.def("append_paged_kv_cache_prefill", &append_paged_kv_cache_prefill);
	m.def("append_paged_kv_cache_decode", &append_paged_kv_cache_decode);
	m.def("estimate_scores", &estimate_scores);
	m.def("select_topk", &select_topk);
	m.def("prefill_select_topk", &prefill_select_topk);
  py::class_<BatchPrefillWithPagedKVCachePyTorchWrapper>(
      m, "BatchPrefillWithPagedKVCachePyTorchWrapper")
      .def(py::init<unsigned int, bool>())
      .def("begin_forward", &BatchPrefillWithPagedKVCachePyTorchWrapper::BeginForward)
      .def("end_forward", &BatchPrefillWithPagedKVCachePyTorchWrapper::EndForward)
      .def("forward", &BatchPrefillWithPagedKVCachePyTorchWrapper::Forward);
  py::class_<BatchDecodeWithPagedKVCachePyTorchWrapper>(
	  m, "BatchDecodeWithPagedKVCachePyTorchWrapper")
      .def(py::init<unsigned int, bool, unsigned int>())
      .def("begin_forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::BeginForward)
      .def("end_forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::EndForward)
      .def("forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::Forward);
	m.def("recall_cuda_knl", &recall_cuda_knl);
	m.def("recall_torch_cpy", &recall_torch_cpy);
	m.def("recall_cuda_cpy", &recall_cuda_cpy);
	m.def("recall_cuda_cpy_cpuhnd_2buf", &recall_cuda_cpy_cpuhnd_2buf);
	m.def("recall_cuda_cpy_cpuhnd_2buf_pool", &recall_cuda_cpy_cpuhnd_2buf_pool);
	m.def("recall_tokens_linear", &recall_tokens_linear);
	m.def("init_recall_thread_pool", &init_recall_thread_pool);
	m.def("shutdown_recall_thread_pool", &shutdown_recall_thread_pool);
	m.def("estimate_select_recall_pool", &estimate_select_recall_pool);
		
	m.def("alloc_managed_bool", &alloc_managed_bool);
	m.def("alloc_managed_bool_scalar", &alloc_managed_bool_scalar);
	m.def("get_corr_managed_cuda", &get_corr_managed_cuda);
}
