from typing import List, Union, Dict, Tuple, Optional
from collections import defaultdict
import os

import torch
from torch import Tensor

from .kv_cache import KvPool, KvCache
from . import kernels
from . import utils
from .echo_token_runtime import EchoTokenPrefetchRuntime
import atexit

Digest = Tuple[Tensor, Tensor]


class InferState:
    def __init__(
        self,
        n_layers,
        n_qo_heads,
        n_kv_heads,
        head_dim,
        page_size,
        dtype: torch.dtype,
        device: torch.device,
        page_budgets: Union[int, List[int]] = None,
        n_max_pages=None,
        n_unlimited_layers=None,
        n_max_bytes=None,
        page_topks: Union[int, List[int]] = None,
        n_max_cpu_pages=None,
        n_max_cpu_bytes=None,
        n_sink_pages=2,
        n_win_pages=2,
        use_sparse_attn=False,
        group_size=None,
        n_groups=None,
        cpu_layout="NHD",
        spec_ret=False,
        thread_pool_size=2,
        n_recall_stream=2,
        recall_impl=None,
        corr=None,
        corr_impl=None,
        corr_max_batch=16,
        **kwargs,
    ) -> None:
        self.n_layers = n_layers
        self.n_qo_heads = n_qo_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.dtype = dtype
        self.device = device

        if n_max_pages is None:
            assert n_max_bytes is not None
            n_max_pages = n_max_bytes // (
                2 * page_size * n_kv_heads * head_dim * dtype.itemsize
            )
        if n_max_cpu_pages is None:
            assert n_max_cpu_bytes is not None
            n_max_cpu_pages = n_max_cpu_bytes // (
                2 * page_size * n_kv_heads * head_dim * dtype.itemsize
            )
        if n_unlimited_layers is None:
            n_unlimited_layers = 0
        if not isinstance(page_budgets, (list, tuple)):
            page_budgets = [None] * n_unlimited_layers + [page_budgets] * (
                n_layers - n_unlimited_layers
            )
        full_attn_layers_env = os.getenv("FULLKV_LAYERS")
        if full_attn_layers_env:
            full_attn_layers = [
                int(x.strip()) for x in full_attn_layers_env.split(",") if x.strip()
            ]
            for layer_idx in full_attn_layers:
                assert 0 <= layer_idx < n_layers, (
                    f"FULL_ATTN_LAYERS has out-of-range layer index {layer_idx}, "
                    f"valid range is [0, {n_layers - 1}]"
                )
                page_budgets[layer_idx] = None
        if page_topks is None:
            page_topks = [b and b // 2 for b in page_budgets]
        elif not isinstance(page_topks, (list, tuple)):
            page_topks = [None] * n_unlimited_layers + [page_topks] * (
                n_layers - n_unlimited_layers
            )

        self.page_size = page_size
        self.n_max_pages = n_max_pages
        self.layer2budget = page_budgets
        self.budget2layers = defaultdict(list)
        for i, b in enumerate(page_budgets):
            self.budget2layers[b].append(i)
        self.layer2topk = page_topks
        self.n_sink_pages = n_sink_pages
        self.n_win_pages = n_win_pages
        assert n_win_pages >= 2
        self.use_sparse_attn = use_sparse_attn

        for i in range(n_layers):
            b = self.layer2budget[i]
            k = self.layer2topk[i]
            if b is not None:
                assert k is not None and k < b
                k = k - n_sink_pages - (n_win_pages - 1)
                assert k > 0
                self.layer2topk[i] = k

        self.layout = "NHD"
        self._i32 = dict(dtype=torch.int32, device=self.device)
        self._u8 = dict(dtype=torch.uint8, device=self.device)
        self._fp = dict(dtype=self.dtype, device=self.device)
        self._bool = dict(dtype=torch.bool, device=self.device)
        self._cpu_bool = dict(dtype=torch.bool, device=torch.device("cpu"))

        self._pool = KvPool(n_max_pages, page_size, n_kv_heads, head_dim, dtype, device)
        self.kv_caches: List[KvCache] = [None] * self.n_layers
        self.dg_caches: List[KvCache] = [None] * self.n_layers
        self._cpu_pool = KvPool(
            n_max_cpu_pages, page_size, n_kv_heads, head_dim, dtype, 
            torch.device("cpu"), layout=cpu_layout
        )
        self.cpu_kv_caches: List[KvCache] = [None] * self.n_layers
        self.topk_dout: Tensor = None
        self.topk_iout: Tensor = None
        self.topk_newi: Tensor = None
        self.topk_rids: Tensor = None
        self.topk_buff: Tensor = None

        self.kv_last_page_len = 0
        self.kv_last_page_lens: Tensor = None
        self.kv_indptrs_tab: Dict[int, Tensor] = {b: None for b in self.budget2layers}
        self.kv_decode_indptrs_tab: Dict[int, Tensor] = {
            b: None for b in self.budget2layers
        }

        self.dg_last_page_len = 0
        self.dg_last_page_lens: Tensor = None
        self.dg_indptrs: Tensor = None

        # maybe larger for larger batch
        wbufs = [torch.empty(64 * 1024 * 1024, **self._u8) for _ in self.budget2layers]
        self.prefill_handler = kernels.BatchPrefillWithPagedKVCacheWrapper(
            wbufs[0], self.layout
        )
        self.prefill_handler_tab = {
            b: kernels.BatchPrefillWithPagedKVCacheWrapper(w, self.layout)
            for b, w in zip(self.budget2layers, wbufs)
        }
        self.decode_handler_tab = {
            b: kernels.BatchDecodeWithPagedKVCacheWrapper(w, self.layout)
            for b, w in zip(self.budget2layers, wbufs)
        }

        self.default_stream = torch.cuda.default_stream(self.device)
        self.prefill_backup_stream = torch.cuda.Stream(self.device)
        self.prefill_backup_events = [None] * self.n_layers
        self.prefill_evicted_pages = [None] * self.n_layers
        self.decode_backup_stream = torch.cuda.Stream(self.device)

        # kv heads within the same group have common selection indices
        assert group_size is None or n_groups is None
        if group_size is None and n_groups is None:
            n_groups = 1
            group_size = n_kv_heads
        elif group_size is None:
            assert n_kv_heads % n_groups == 0
            group_size = n_kv_heads // n_groups
        else:
            assert n_kv_heads % group_size == 0
            n_groups = n_kv_heads // group_size
        self.group_size = group_size
        self.n_groups = n_groups

        self.sel_policy = kwargs.get("sel_policy", "topk")
        assert self.sel_policy in ("topk", "echokv_token"), (
            f"Unknown sel_policy: {self.sel_policy}"
        )
        self.echo_token = self.sel_policy == "echokv_token"
        self.echo_num_anchors = int(kwargs.get("echo_num_anchors", 64))
        self.echo_shared_batch = bool(kwargs.get("echo_shared_batch", True))
        self.echo_use_cuda_token_recall = bool(
            kwargs.get("echo_use_cuda_token_recall", True)
        )
        self.profile_timing = bool(kwargs.get("profile_timing", True))
        assert self.echo_num_anchors > 0

        assert recall_impl in ("arkvale", "torch_cpy", "cuda_cpy"), f"Unknown recall_impl: {recall_impl}"
        self.recall_impl = recall_impl
        self.recall_buf1 = torch.empty((0), **self._fp)
        self.recall_buf2 = torch.empty((0), **self._fp)
        if page_budgets[-1] is not None and cpu_layout == "HND":
            self.recall_buf1 = torch.empty((page_budgets[-1], group_size, 2, page_size, head_dim), 
                                           **self._fp)
            self.recall_buf2 = torch.empty((page_budgets[-1], group_size, 2, page_size, head_dim), 
                                           **self._fp)
        
        assert n_recall_stream % 2 == 0
        self.n_recall_stream = n_recall_stream
        self.recall_streams = [torch.cuda.Stream(self.device) 
                                for _ in range(n_recall_stream)]

        self.spec_ret = spec_ret
        self.corr_impl = corr_impl or os.getenv("CORR_IMPL", "managed_cuda")
        assert self.corr_impl in ("torch", "managed_cuda")
        self.corr_max_batch = corr_max_batch
        assert self.corr_max_batch > 0
        self.to_corr_gpu = torch.empty((self.corr_max_batch, self.n_kv_heads), **self._bool)
        self.need_corr_gpu = torch.empty((1,), **self._bool)
        self.to_corr_cpu_pinned = torch.empty(
            (self.corr_max_batch, self.n_kv_heads), pin_memory=True, **self._cpu_bool
        )
        self.need_corr_cpu_pinned = torch.empty((1,), pin_memory=True, **self._cpu_bool)
        self.to_corr_managed = None
        if self.corr_impl == "managed_cuda":
            self.to_corr_managed = kernels.alloc_managed_bool(
                self.corr_max_batch, self.n_kv_heads
            )
        if spec_ret:
            assert self.recall_impl == "cuda_cpy" and cpu_layout == "HND", f"{self.recall_impl=} {cpu_layout=}"
            self.compute_stream = torch.cuda.Stream(self.device) 
            self.spec_ret_recall_status = [None] * n_layers
            self.spec_ret_recall_events = [
                (torch.cuda.Event(), torch.cuda.Event()) for _ in range(n_layers)
            ]
            for e1, e2 in self.spec_ret_recall_events:
                # init to be a valid event...
                e1.record()
                e2.record()
            try:
                kernels.init_recall_thread_pool(thread_pool_size)
                atexit.register(self._shutdown_cpp_pool)
            except Exception as e:
                raise RuntimeError("cannot initialize thread pool") from e
            self.corr = corr
            self.last_step_q = [None] * n_layers
        else:
            self.compute_stream = self.default_stream

        self.recall_stat_ms = []
        self.sel_stat_ms = []
        self.attn_stat_ms = []
        self._perf_tot_ms = {
            "select": 0.0,
            "recall": 0.0,
            "attn": 0.0,
        }
        self._perf_pending = {
            "select": [],
            "recall": [],
            "attn": [],
        }
        self.corr_checks = [0] * n_layers
        self.corr_triggers = [0] * n_layers

        self.echo_runtimes: List[Optional[EchoTokenPrefetchRuntime]] = [None] * n_layers
        if self.echo_token:
            for i in range(self.n_layers):
                budget = self.layer2budget[i]
                if budget is None:
                    continue
                mid_pages = max(0, self.layer2topk[i] or 0)
                self.echo_runtimes[i] = EchoTokenPrefetchRuntime(
                    n_qo_heads=self.n_qo_heads,
                    n_kv_heads=self.n_kv_heads,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    n_sink_pages=self.n_sink_pages,
                    n_win_pages=self.n_win_pages,
                    mid_pages=mid_pages,
                    seed_anchors=self.echo_num_anchors,
                    shared_batch=self.echo_shared_batch,
                    use_cuda_token_recall=self.echo_use_cuda_token_recall,
                )
    
    def _shutdown_cpp_pool(self):
        try:
            kernels.shutdown_recall_thread_pool()
        except Exception as e:
            raise RuntimeError("cannot shutdown thread pool") from e


    def get_corr_trigger_stats(self):
        total_checks = sum(self.corr_checks)
        total_triggers = sum(self.corr_triggers)
        total_rate = (total_triggers / total_checks) if total_checks > 0 else 0.0
        return {
            "total_checks": total_checks,
            "total_triggers": total_triggers,
            "total_rate": total_rate,
            "layer_checks": self.corr_checks,
            "layer_triggers": self.corr_triggers,
        }

    def _perf_start(self, section: str, stream: torch.cuda.Stream = None):
        if (not self.profile_timing) or section not in self._perf_pending:
            return None
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        evt = torch.cuda.Event(enable_timing=True)
        evt.record(stream)
        return evt

    def _perf_stop(self, section: str, start_evt, stream: torch.cuda.Stream = None):
        if (not self.profile_timing) or start_evt is None:
            return
        if section not in self._perf_pending:
            return
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        end_evt = torch.cuda.Event(enable_timing=True)
        end_evt.record(stream)
        self._perf_pending[section].append((start_evt, end_evt))

    def flush_perf_stats(self, synchronize: bool = True):
        if not self.profile_timing:
            return
        if synchronize:
            torch.cuda.synchronize(self.device)
        sec2arr = {
            "select": self.sel_stat_ms,
            "recall": self.recall_stat_ms,
            "attn": self.attn_stat_ms,
        }
        for sec, pairs in self._perf_pending.items():
            if not pairs:
                continue
            arr = sec2arr[sec]
            acc = self._perf_tot_ms[sec]
            for s, e in pairs:
                ms = float(s.elapsed_time(e))
                arr.append(ms)
                acc += ms
            self._perf_tot_ms[sec] = acc
            pairs.clear()

    def reset_perf_stats(self):
        self.sel_stat_ms.clear()
        self.recall_stat_ms.clear()
        self.attn_stat_ms.clear()
        for k in self._perf_tot_ms:
            self._perf_tot_ms[k] = 0.0
        for k in self._perf_pending:
            self._perf_pending[k].clear()

    def get_perf_stats(self, synchronize: bool = True, reset: bool = False):
        self.flush_perf_stats(synchronize=synchronize)
        out = {
            "select_ms": float(self._perf_tot_ms["select"]),
            "recall_ms": float(self._perf_tot_ms["recall"]),
            "attn_ms": float(self._perf_tot_ms["attn"]),
            "select_calls": len(self.sel_stat_ms),
            "recall_calls": len(self.recall_stat_ms),
            "attn_calls": len(self.attn_stat_ms),
        }
        if reset:
            self.reset_perf_stats()
        return out

    @property
    def seq_len(self):
        return self.kv_caches[0].seq_len

    @property
    def n_pages(self):
        return self.kv_caches[0].n_pages

    @property
    def batch_size(self):
        return self.kv_caches[0].batch_size

    def _prepare_prefill(self, bsz, q_len):
        if self.echo_token:
            for rt in self.echo_runtimes:
                if rt is None:
                    continue
                rt.ensure(bsz, self.device, self.dtype)
                rt.reset_state()
        self._pool.clear()
        self.kv_caches = [
            KvCache(
                self._pool,
                bsz,
                self.layer2budget[i],
                self.n_sink_pages,
                self.n_win_pages,
                n_groups=self.n_groups,
            )
            for i in range(self.n_layers)
        ]
        self.dg_caches = [
            self.layer2budget[i] and KvCache(self._pool, bsz)
            for i in range(self.n_layers)
        ]
        self._cpu_pool.clear()
        self.cpu_kv_caches = [
            self.layer2budget[i] and KvCache(self._cpu_pool, bsz)
            for i in range(self.n_layers)
        ]

        max_topk = max([k for k in self.layer2topk if k] + [0])
        max_cap = max([b for b in self.layer2budget if b] + [0])
        bsz1 = bsz * self.n_groups
        self.topk_dout = torch.empty([self.n_layers, bsz1 * max_cap], **self._fp)
        self.topk_iout = torch.empty(
            [self.n_layers, bsz1 * (max_topk + self.n_sink_pages + self.n_win_pages)], **self._i32
        )
        self.topk_newi = torch.empty([self.n_layers, bsz1 * max_cap], **self._fp)
        self.topk_rids = torch.empty([self.n_layers, bsz1 * (max_topk + 1)], **self._i32)

        self.topk_buff = torch.empty([self.n_layers, bsz1 * (1 << 10)], **self._u8)
        # we do not pre-allocate real kv-pages before prefill
        [kvc.prefill_alloc_n_tokens(q_len) for kvc in self.cpu_kv_caches if kvc]
        n_kv_pages = (q_len + self.page_size - 1) // self.page_size
        self.kv_last_page_len = (q_len - 1) % self.page_size + 1
        self.kv_last_page_lens = torch.tensor(
            [self.kv_last_page_len] * bsz, **self._i32
        )
        for b in self.kv_indptrs_tab:
            self.kv_indptrs_tab[b] = kv_indptr = torch.arange(
                0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
            )

        n_filled_kv_pages = n_kv_pages - 1
        assert n_filled_kv_pages > 0, (
            f"prefill requires q_len > page_size for digest cache construction: "
            f"q_len={q_len}, page_size={self.page_size}"
        )
        [
            dgc.prefill_alloc_n_tokens(n_filled_kv_pages, self.alloc_page)
            for dgc in self.dg_caches
            if dgc
        ]
        n_dg_pages = (n_filled_kv_pages + self.page_size - 1) // self.page_size
        self.dg_last_page_len = (n_filled_kv_pages - 1) % self.page_size + 1
        self.dg_last_page_lens = torch.tensor(
            [self.dg_last_page_len] * bsz, **self._i32
        )
        self.dg_indptrs = torch.arange(0, bsz * n_dg_pages + 1, n_dg_pages, **self._i32)

        self.rope_indptr = torch.arange(0, (bsz+1)*q_len, q_len, **self._i32)
        self.rope_offsets = torch.zeros(bsz, **self._i32)

        qo_indptr = torch.arange(0, bsz * q_len + 1, q_len, **self._i32)
        self.prefill_handler.begin_forward(
            qo_indptr,
            kv_indptr,
            self.kv_last_page_lens,
            self.n_qo_heads,
            self.n_kv_heads,
            self.head_dim,
            self.page_size,
            self.dtype,
        )

    def _finish_prefill(self, bsz, q_len):
        for b, ls in self.budget2layers.items():
            n_kv_pages = utils.all_eq(self.kv_caches[l].n_real_pages for l in ls)
            self.kv_decode_indptrs_tab[b] = self.kv_indptrs_tab[b] = torch.arange(
                0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
            )
            if b is not None and self.use_sparse_attn:
                topk = utils.all_eq(self.layer2topk[l] for l in ls)
                n_kv_pages = min(
                    self.n_sink_pages + topk + self.n_win_pages, n_kv_pages
                )
                self.kv_decode_indptrs_tab[b] = torch.arange(
                    0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
                )
        self.prefill_handler.end_forward()

    def _prepare_decode(self, bsz):
        if self.kv_last_page_len + 1 >= self.page_size:
            self.default_stream.wait_stream(self.decode_backup_stream)
        pre = [kvc.n_real_pages for kvc in self.kv_caches]
        n_new_kv_pages = utils.all_eq(
            kvc.decode_alloc_1_token(self.alloc_page) for kvc in self.kv_caches
        )
        self.kv_last_page_len = utils.all_eq(
            kvc.last_page_len for kvc in self.kv_caches
        )
        self.kv_last_page_lens = torch.tensor(
            [self.kv_last_page_len] * bsz, **self._i32
        )
        [kvc.decode_alloc_1_token() for kvc in self.cpu_kv_caches if kvc]

        if n_new_kv_pages > 0:
            assert n_new_kv_pages == 1

            cur = [kvc.n_real_pages for kvc in self.kv_caches]
            for b, ls in self.budget2layers.items():
                n_new_kv_real_pages = utils.all_eq(cur[l] - pre[l] for l in ls)
                if n_new_kv_real_pages > 0:
                    n_kv_pages = self.kv_caches[ls[0]].n_real_pages
                    self.kv_decode_indptrs_tab[b] = self.kv_indptrs_tab[b] = (
                        torch.arange(0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32)
                    )
                    if b is not None and self.use_sparse_attn:
                        topk = utils.all_eq(self.layer2topk[l] for l in ls)
                        n_kv_pages = min(
                            self.n_sink_pages + topk + self.n_win_pages, n_kv_pages
                        )
                        self.kv_decode_indptrs_tab[b] = torch.arange(
                            0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
                        )

            dg_caches = [dgc for dgc in self.dg_caches if dgc]
            if len(dg_caches) > 0:
                n_new_dg_pages = utils.all_eq(
                    dgc.decode_alloc_1_token(self.alloc_page) for dgc in dg_caches
                )
                self.dg_last_page_len = utils.all_eq(
                    dgc.last_page_len for dgc in dg_caches
                )
                self.dg_last_page_lens = torch.tensor(
                    [self.dg_last_page_len] * bsz, **self._i32
                )
                if n_new_dg_pages > 0:
                    assert n_new_dg_pages == 1
                    n_dg_pages = utils.all_eq(dgc.n_real_pages for dgc in dg_caches)
                    self.dg_indptrs = torch.arange(
                        0, bsz * n_dg_pages + 1, n_dg_pages, **self._i32
                    )

                with torch.cuda.stream(self.decode_backup_stream):
                    [self.decode_backup_1_page(l) for l in range(self.n_layers)]
                [self.decode_save_1_digest(l) for l in range(self.n_layers)]

        self.rope_indptr = torch.arange(0, bsz+1, 1, **self._i32)
        self.rope_offsets = torch.full((bsz,), self.seq_len, **self._i32)

        for b, h in self.decode_handler_tab.items():
            h.begin_forward(
                self.kv_decode_indptrs_tab[b],
                self.kv_last_page_lens,
                self.n_qo_heads,
                self.n_kv_heads,
                self.head_dim,
                self.page_size,
                data_type=self.dtype,
            )

    def _finish_decode(self, bsz):
        for handler in self.decode_handler_tab.values():
            handler.end_forward()

    def begin_forward(self, bsz, q_len):
        if q_len > 1:
            self._prepare_prefill(bsz, q_len)
        else:
            self._prepare_decode(bsz)

    def end_forward(self, bsz, q_len):
        if q_len > 1:
            self._finish_prefill(bsz, q_len)
        else:
            self._finish_decode(bsz)

    def append_paged_kv_cache(self, layer_idx: int, keys: Tensor, vals: Tensor):
        kvc = self.kv_caches[layer_idx]
        kernels.append_paged_kv_cache(
            keys,
            vals,
            kvc.buffer,
            kvc.c2p,
            self.kv_indptrs_tab[self.layer2budget[layer_idx]],
            self.kv_last_page_lens,
            self.layout,
        )

    def save_digests(self, layer_idx: int, digest: Digest):
        dgc = self.dg_caches[layer_idx]
        kernels.append_paged_kv_cache(
            *digest,
            dgc.buffer,
            dgc.c2p,
            self.dg_indptrs,
            self.dg_last_page_lens,
            self.layout,
        )

    def estimate_scores(
        self, layer_idx: int, query_states: Tensor, n_groups: int = None
    ):
        if n_groups is None:
            n_groups = self.n_groups
        dgc = self.dg_caches[layer_idx]
        return kernels.estimate_scores(
            query_states,
            dgc.buffer,
            dgc.c2p,
            self.dg_indptrs,
            self.dg_last_page_lens,
            dgc.seq_len,
            self.layout,
            n_groups,
        )

    def select_topk(self, layer_idx: int, scores: Tensor):
        bsz = self.batch_size * self.n_groups
        budget = self.layer2budget[layer_idx]
        topk = self.layer2topk[layer_idx]
        kvc = self.kv_caches[layer_idx]
        ns = self.n_sink_pages
        nw = self.n_win_pages
        eids_range = topk + ns + nw
        dout = self.topk_dout[layer_idx, : bsz * topk].view(bsz, topk)
        eids = self.topk_iout[layer_idx, : bsz * eids_range].view(bsz, eids_range)
        newi = self.topk_newi[layer_idx, : bsz * budget].view(bsz, budget)
        rids = self.topk_rids[layer_idx, : bsz * (topk + 1)].view(bsz, topk + 1)
        buff = self.topk_buff[layer_idx]
        scores = scores.reshape(bsz, -1)
        cc2gp = kvc.cc2gp.reshape(bsz, -1)
        gc2cc = kvc.gc2cc.reshape(bsz, -1)
        kernels.select_topk(
            scores, dout, eids, newi, cc2gp, gc2cc, rids, buff, topk, ns, nw
        )
        return eids, rids

    def echo_token_on_prefill(
        self,
        layer_idx: int,
        q_last: Tensor,   # [bsz, 1, n_qo_heads, head_dim]
        k_full: Tensor,   # [bsz, q_len, n_kv_heads, head_dim]
        v_full: Tensor,   # [bsz, q_len, n_kv_heads, head_dim]
    ):
        if not self.echo_token:
            return
        rt = self.echo_runtimes[layer_idx]
        if rt is None:
            return
        rt.ensure(q_last.shape[0], self.device, self.dtype)
        rt.copy_prefill_to_cpu(k_full, v_full)
        sel_t0 = self._perf_start("select")
        rt.init_anchors_from_prefill(q_last, k_full)
        self._perf_stop("select", sel_t0)
        starts = rt.build_starts(rt.cpu_len + 1)
        if starts is not None:
            rec_t0 = self._perf_start("recall", stream=rt.recall_stream)
            rt.launch_prefetch(starts, target_seq_len=rt.cpu_len + 1)
            self._perf_stop("recall", rec_t0, stream=rt.recall_stream)

    def echo_token_on_decode_append(
        self,
        layer_idx: int,
        k_tok: Tensor,    # [bsz, 1, n_kv_heads, head_dim]
        v_tok: Tensor,    # [bsz, 1, n_kv_heads, head_dim]
    ):
        if not self.echo_token:
            return
        rt = self.echo_runtimes[layer_idx]
        if rt is None:
            return
        rt.ensure(k_tok.shape[0], self.device, self.dtype)
        rt.append_decode_token_to_cpu(k_tok, v_tok)

    def decode_echo_token(self, layer_idx: int, q: Tensor):
        # q: [bsz, 1, n_qo_heads, head_dim]
        rt = self.echo_runtimes[layer_idx]
        assert rt is not None
        cur_seq = rt.cpu_len
        starts = rt.build_starts(cur_seq)
        if starts is not None and (
            (not rt.prefetch_ready) or (rt.pending_seq_len != int(cur_seq))
        ):
            rec_t0 = self._perf_start("recall", stream=rt.recall_stream)
            rt.launch_prefetch(starts, target_seq_len=cur_seq)
            self._perf_stop("recall", rec_t0, stream=rt.recall_stream)

        ok = rt.activate_prefetch(cur_seq, self.compute_stream)
        if not ok:
            # Fallback: no sparse-ready region, use dense paged decode.
            return self.decode_sdpa(layer_idx, q, self.kv_caches[layer_idx].c2p)

        local_k, local_v, k_mid = rt.local_kv(q.shape[0])
        attn_t0 = self._perf_start("attn")
        out = rt.attend(q, local_k, local_v)
        self._perf_stop("attn", attn_t0)

        sel_t0 = self._perf_start("select")
        rt.update_anchors_from_middle(q, k_mid)
        self._perf_stop("select", sel_t0)

        next_starts = rt.build_starts(cur_seq + 1)
        if next_starts is not None:
            rec_t1 = self._perf_start("recall", stream=rt.recall_stream)
            rt.launch_prefetch(next_starts, target_seq_len=cur_seq + 1)
            self._perf_stop("recall", rec_t1, stream=rt.recall_stream)
        return out

    def recall(self, layer_idx: int, 
               eids: Tensor, rids: Tensor, 
               blocking=True,
               recall_evt1=None, 
               recall_evt2=None,
               need_recall_corr=None,):
        nw = self.n_win_pages
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        bsz = kvc.batch_size
        gs = self.group_size
        ng = self.n_groups
        eids = eids.reshape(bsz, ng, -1)
        rids = rids.reshape(bsz, ng, -1)
        impl = self.recall_impl

        torch.cuda.nvtx.range_push("recall")
        rec_stream = torch.cuda.current_stream(self.device)
        rec_t0 = self._perf_start("recall", stream=rec_stream)
        if impl == "arkvale":
            for i in range(kvc.batch_size):
                for j in range(self.n_groups):
                    nr = rids[i, j, 0]
                    if nr == 0:
                        continue
                    heads = slice(j * gs, (j + 1) * gs)
                    if cpu_kvc.layout == "NHD":
                        for ei, ri in zip(eids[i, j, -(nr + nw) : -nw], rids[i, j, 1 : nr + 1]):
                            kvc.pool[ei][..., heads, :].copy_(
                                cpu_kvc[i, ri][..., heads, :], non_blocking=True
                            )
                    else:
                        for ei, ri in zip(eids[i, j, -(nr + nw) : -nw], rids[i, j, 1 : nr + 1]):
                            kvc.pool[ei][..., heads, :].copy_(
                                cpu_kvc[i, ri][heads, ...].view(2, self.page_size, gs, -1), 
                                non_blocking=True
                            )
            torch.cuda.synchronize(self.device)
        else:
            if impl == "cuda_cpy" and cpu_kvc.layout == "HND":
                # NOTE: need_recall only valid here
                rs_group = layer_idx % (self.n_recall_stream // 2)
                recall_stream1 = self.recall_streams[rs_group*2]
                recall_stream2 = self.recall_streams[rs_group*2 + 1]
                if blocking:
                    kernels.recall_cuda_cpy_cpuhnd_2buf(
                        rids, eids, cpu_kvc.c2p, cpu_kvc.buffer, kvc.buffer, 
                        ng, gs, nw, self.recall_buf1, self.recall_buf2,
                        recall_stream1.cuda_stream, recall_stream2.cuda_stream,
                        need_recall_corr
                    )
                else:
                    # non-blocking double buffer
                    kernels.recall_cuda_cpy_cpuhnd_2buf_pool(
                        rids, eids, cpu_kvc.c2p, cpu_kvc.buffer, kvc.buffer, 
                        ng, gs, nw, self.recall_buf1, self.recall_buf2,
                        recall_stream1.cuda_stream,
                        recall_stream2.cuda_stream,
                        recall_evt1.cuda_event,
                        recall_evt2.cuda_event,
                        need_recall_corr
                    )
            else:
                kernels.recall(rids, eids, cpu_kvc.c2p, cpu_kvc.buffer, kvc.buffer, 
                                ng, gs, nw, impl, cpu_kvc.layout, self.recall_buf1)
        self._perf_stop("recall", rec_t0, stream=rec_stream)
        torch.cuda.nvtx.range_pop()

    def estimate_select_recall(self, layer_idx: int, q: Tensor):
        torch.cuda.nvtx.range_push("score&topk")
        sel_t0 = self._perf_start("select")
        scores = self.estimate_scores(layer_idx, q)
        eids, rids = self.select_topk(layer_idx, scores)
        self._perf_stop("select", sel_t0)
        torch.cuda.nvtx.range_pop()

        self.recall(layer_idx, eids, rids, blocking=True, need_recall_corr=torch.Tensor())

    def estimate_select(self, layer_idx: int, q: Tensor):
        torch.cuda.nvtx.range_push("score&topk")
        sel_t0 = self._perf_start("select")
        scores = self.estimate_scores(layer_idx, q)
        eids, rids = self.select_topk(layer_idx, scores)
        self._perf_stop("select", sel_t0)
        torch.cuda.nvtx.range_pop()
        return eids, rids

    def estimate_select_recall_pool(self, layer_idx: int, q: Tensor, 
                                    recall_evt1: torch.cuda.Event, 
                                    recall_evt2: torch.cuda.Event):
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        dgc = self.dg_caches[layer_idx]
        bsz = self.batch_size * self.n_groups
        budget = self.layer2budget[layer_idx]
        topk = self.layer2topk[layer_idx]
        ng = self.n_groups
        gs = self.group_size
        ns = self.n_sink_pages
        nw = self.n_win_pages
        eids_range = topk + ns + nw
        dout = self.topk_dout[layer_idx, : bsz * topk].view(bsz, topk)
        eids = self.topk_iout[layer_idx, : bsz * eids_range].view(bsz, eids_range)
        newi = self.topk_newi[layer_idx, : bsz * budget].view(bsz, budget)
        rids = self.topk_rids[layer_idx, : bsz * (topk + 1)].view(bsz, topk + 1)
        buff = self.topk_buff[layer_idx]
        cc2gp = kvc.cc2gp.reshape(bsz, -1)
        gc2cc = kvc.gc2cc.reshape(bsz, -1)

        rs_group = layer_idx % (self.n_recall_stream // 2)
        recall_stream1 = self.recall_streams[rs_group*2]
        recall_stream2 = self.recall_streams[rs_group*2 + 1]
        rec_t0 = self._perf_start("recall", stream=recall_stream1)
        kernels.estimate_select_recall_pool(
            # for estimate
            q, dgc.buffer, dgc.c2p, dgc.seq_len,
            self.dg_indptrs, self.dg_last_page_lens,
            # for topk
            dout, eids, rids, newi, buff,
            cc2gp, gc2cc, topk,
            # for recall
            cpu_kvc.c2p, cpu_kvc.buffer,
            kvc.buffer,
            ng, gs, ns, nw, 
            self.recall_buf1, self.recall_buf2,
            recall_stream1.cuda_stream, recall_stream2.cuda_stream,
            recall_evt1.cuda_event, recall_evt2.cuda_event
        )
        self._perf_stop("recall", rec_t0, stream=recall_stream1)

    def prefill_backup_pages(self, layer_idx: int):
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        if kvc.budget is None:
            return
        for ci, gi in zip(cpu_kvc.c2p.reshape(-1), kvc.c2p.reshape(-1)):
            if cpu_kvc.layout == "HND":
                # [2, page_size, n_kv_heads, head_dim] => [n_kv_heads, 2, page_size, head_dim]
                cpu_kvc.buffer[ci].copy_(kvc.buffer[gi].permute(2, 0, 1, 3).contiguous(),
                                         non_blocking=True)
            else:
                cpu_kvc.buffer[ci].copy_(kvc.buffer[gi], non_blocking=True)

    def decode_backup_1_page(self, layer_idx: int):
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        if kvc.budget is None:
            return
        for i in range(kvc.batch_size):
            if cpu_kvc.layout == "HND":
                cpu_kvc[i, -2].copy_(kvc[i, kvc.evict_idx].permute(2, 0, 1, 3).contiguous(), 
                                     non_blocking=True)
            else:
                cpu_kvc[i, -2].copy_(kvc[i, kvc.evict_idx], non_blocking=True)

    def _summarize_keys(self, filled_keys: Tensor) -> Digest:
        maxs = filled_keys.max(dim=2).values
        mins = filled_keys.min(dim=2).values
        centers = (maxs + mins) / 2
        dists = (
            (
                centers.reshape(*filled_keys.shape[:2], 1, -1, self.head_dim)
                - filled_keys
            )
            .abs()
            .mean(dim=2)
        )
        mins = centers - dists
        maxs = centers + dists
        return maxs, mins

    def prefill_save_digests(self, layer_idx: int, keys: Tensor):
        if self.layer2budget[layer_idx] is None:
            return
        bsz, q_len, *_ = keys.shape
        n_filled_pages = (q_len + self.page_size - 1) // self.page_size - 1
        filled_keys = keys[:, : n_filled_pages * self.page_size, ...].reshape(
            bsz, n_filled_pages, self.page_size, self.n_kv_heads, self.head_dim
        )
        self.save_digests(layer_idx, self._summarize_keys(filled_keys))

    def decode_save_1_digest(self, layer_idx: int):
        kvc = self.kv_caches[layer_idx]
        if kvc.budget is None:
            return
        filled_keys = torch.cat(
            [kvc[i, kvc.evict_idx][:1][None, ...] for i in range(self.batch_size)],
            dim=0,
        )
        self.save_digests(layer_idx, self._summarize_keys(filled_keys))

    def prefill_evict_extra_pages(self, layer_idx: int, query_states: Tensor):
        kvc = self.kv_caches[layer_idx]
        bsz = kvc.batch_size
        ng = self.n_groups
        if kvc.budget is None:
            return
        if kvc.n_real_pages > kvc.budget:
            ns = max(2, kvc.n_sink_pages)
            nw = kvc.n_win_pages
            assert ns + nw <= kvc.budget
            topk = kvc.budget - ns - nw
            kvc.c2p = torch.empty([bsz, kvc.budget], **kvc._i32)
            kvc.gc2cc = torch.empty([bsz, kvc.budget], **kvc._i32)
            ev_gpi = kvc.cc2gp.clone()
            kvc.cc2gp.fill_(-1)
            dout = self.topk_dout[layer_idx, : bsz * topk].view(bsz, topk)
            buff = self.topk_buff[layer_idx]
            scores = self.estimate_scores(layer_idx, query_states, n_groups=1)
            scores = scores.reshape(bsz, -1)
            kernels.prefill_select_topk(
                scores, dout, kvc.c2p, kvc.cc2gp, ev_gpi, kvc.gc2cc, buff, topk, ns, nw
            )
            self.prefill_evicted_pages[layer_idx] = ev_gpi
        kvc.cc2gp = (
            kvc.cc2gp.reshape(bsz, 1, -1)
            .expand(bsz, ng, kvc.cc2gp.shape[-1])
            .contiguous()
        )
        kvc.gc2cc = (
            kvc.gc2cc.reshape(bsz, 1, -1)
            .expand(bsz, ng, kvc.gc2cc.shape[-1])
            .contiguous()
        )

    def alloc_page(self):
        if len(self._pool._free_ids) <= 0:
            for i in range(self.n_layers):
                kvc = self.kv_caches[i]
                evt = self.prefill_backup_events[i]
                ev_gpi = self.prefill_evicted_pages[i]
                if ev_gpi is None:
                    continue
                assert evt is not None
                self.default_stream.wait_event(evt)
                [
                    kvc.pool.free_page(pid)
                    for pid in ev_gpi.reshape(-1).tolist()
                    if pid >= 0
                ]
                self.prefill_backup_events[i] = None
                self.prefill_evicted_pages[i] = None
                break
        return self._pool.alloc_page()

    def prefill_sdpa(self, layer_idx: int, q: Tensor, page_ids: Tensor = None):
        kvc = self.kv_caches[layer_idx]
        if page_ids is None:
            page_ids = kvc.c2p
        return self.prefill_handler.forward(q, kvc.buffer, page_ids)

    def decode_sdpa(self, layer_idx: int, q: Tensor, page_ids: Tensor = None):
        kvc = self.kv_caches[layer_idx]
        if page_ids is None:
            page_ids = kvc.c2p
        attn_t0 = self._perf_start("attn")
        out = self.decode_handler_tab[kvc.budget].forward(q, kvc.buffer, page_ids)
        self._perf_stop("attn", attn_t0)
        return out
