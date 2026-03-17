import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import kernels


class EchoTokenPrefetchRuntime:
    """
    Token-wise EchoKV runtime:
    - CPU linear KV storage (pinned)
    - fixed-size GPU middle buffer
    - async prefetch on dedicated CUDA stream
    - page-aligned tiled starts to fill middle budget
    """

    def __init__(
        self,
        n_qo_heads: int,
        n_kv_heads: int,
        head_dim: int,
        page_size: int,
        n_sink_pages: int,
        n_win_pages: int,
        mid_pages: int,
        seed_anchors: int = 64,
        shared_batch: bool = True,
        use_cuda_token_recall: bool = True,
        anchor_head_sample: int = 0,
    ) -> None:
        self.n_qo_heads = n_qo_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.n_sink_pages = n_sink_pages
        self.n_win_pages = n_win_pages
        self.mid_pages = int(max(0, mid_pages))
        self.mid_tokens = self.mid_pages * self.page_size
        self.seed_anchors = int(max(1, seed_anchors))
        self.shared_batch = bool(shared_batch)
        self.use_cuda_token_recall = bool(use_cuda_token_recall)
        self.anchor_head_sample = int(anchor_head_sample)
        self.n_q_per_kv = self.n_qo_heads // self.n_kv_heads
        self._inv_sqrt_head_dim = 1.0 / math.sqrt(self.head_dim)
        self.delta_max_pages = 8

        self.sink_tokens = self.n_sink_pages * self.page_size
        self.win_tokens = self.n_win_pages * self.page_size
        self.local_pages = self.n_sink_pages + self.mid_pages + self.n_win_pages

        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        self.recall_stream: Optional[torch.cuda.Stream] = None
        self.prefetch_event = None
        self.append_event = None

        self.batch_size = 0
        self.eff_batch = 0
        self.cpu_capacity = 0
        self.cpu_len = 0
        self.cpu_kv = None  # [bsz, cap, 2, n_kv_heads, head_dim], pinned

        self.sink_k = None  # [bsz, n_kv_heads, sink_tokens, head_dim]
        self.sink_v = None
        self.win_k = None   # [bsz, n_kv_heads, win_tokens, head_dim]
        self.win_v = None
        self.sink_len = 0
        self.win_len = 0
        self.win_ptr = 0

        self.local_k_buf = [None, None]  # [eff, n_kv_heads, total, head_dim]
        self.local_v_buf = [None, None]
        self.local_mid_ready = [False, False]
        self.local_total_cap = self.sink_tokens + self.mid_tokens + self.win_tokens
        self.local_sink_len_cached = [0, 0]
        self.page_kv_buf = [None, None]  # [eff, local_pages, 2, page_size, n_kv_heads, head_dim]
        self.page_mid_ready = [False, False]
        self.page_sink_ready = [False, False]
        self.page_win_ready = [False, False]
        self.page_ids = None  # [bsz, local_pages], int32
        self.page_ids_bsz = 0

        self.gpu_mid = [None, None]  # [eff_bsz, mid_tokens, 2, n_kv_heads, head_dim]
        self.pending_idx = 0
        self.active_idx = 0
        self.pending_seq_len = -1
        self.active_seq_len = -1
        self.pending_starts = None
        self.active_starts = None
        self.prefetch_ready = False

        self.anchors = None  # [eff_bsz, mid_pages], token indices
        self._head_idx = None

    def _effective_batch(self, bsz: int) -> int:
        if self.shared_batch and bsz > 1:
            return 1
        return bsz

    def ensure(self, bsz: int, device: torch.device, dtype: torch.dtype):
        if self.mid_pages <= 0:
            self.batch_size = bsz
            self.eff_batch = self._effective_batch(bsz)
            self.device = device
            self.dtype = dtype
            if self.recall_stream is None:
                self.recall_stream = torch.cuda.Stream(device)
                self.prefetch_event = torch.cuda.Event()
                self.append_event = torch.cuda.Event()
            elif self.append_event is None:
                self.append_event = torch.cuda.Event()
            return

        eff_bsz = self._effective_batch(bsz)
        need_alloc = (
            self.device != device
            or self.dtype != dtype
            or self.batch_size != bsz
            or self.eff_batch != eff_bsz
            or self.cpu_kv is None
            or self.gpu_mid[0] is None
        )
        if not need_alloc:
            return

        self.device = device
        self.dtype = dtype
        self.batch_size = bsz
        self.eff_batch = eff_bsz

        if self.recall_stream is None:
            self.recall_stream = torch.cuda.Stream(device)
            self.prefetch_event = torch.cuda.Event()
            self.append_event = torch.cuda.Event()
        elif self.append_event is None:
            self.append_event = torch.cuda.Event()

        min_cap = self.sink_tokens + self.win_tokens + self.mid_tokens + 16
        self.cpu_capacity = max(2048, min_cap)
        self.cpu_kv = torch.empty(
            (bsz, self.cpu_capacity, 2, self.n_kv_heads, self.head_dim),
            dtype=dtype,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        self.gpu_mid[0] = torch.empty(
            (eff_bsz, self.mid_tokens, 2, self.n_kv_heads, self.head_dim),
            dtype=dtype,
            device=device,
        )
        self.gpu_mid[1] = torch.empty_like(self.gpu_mid[0])
        if self.local_total_cap > 0:
            self.local_k_buf[0] = torch.empty(
                (eff_bsz, self.n_kv_heads, self.local_total_cap, self.head_dim),
                dtype=dtype,
                device=device,
            )
            self.local_v_buf[0] = torch.empty_like(self.local_k_buf[0])
            self.local_k_buf[1] = torch.empty_like(self.local_k_buf[0])
            self.local_v_buf[1] = torch.empty_like(self.local_v_buf[0])
        else:
            self.local_k_buf = [None, None]
            self.local_v_buf = [None, None]
        if self.local_pages > 0:
            self.page_kv_buf[0] = torch.empty(
                (
                    eff_bsz,
                    self.local_pages,
                    2,
                    self.page_size,
                    self.n_kv_heads,
                    self.head_dim,
                ),
                dtype=dtype,
                device=device,
            )
            self.page_kv_buf[1] = torch.empty_like(self.page_kv_buf[0])
        else:
            self.page_kv_buf = [None, None]
        self.reset_state()

    def reset_state(self):
        self.cpu_len = 0
        self.sink_k = None
        self.sink_v = None
        self.win_k = None
        self.win_v = None
        self.sink_len = 0
        self.win_len = 0
        self.win_ptr = 0
        self.anchors = None
        self.pending_idx = 0
        self.active_idx = 0
        self.pending_seq_len = -1
        self.active_seq_len = -1
        self.pending_starts = None
        self.active_starts = None
        self.prefetch_ready = False
        self.local_mid_ready = [False, False]
        self.local_sink_len_cached = [0, 0]
        self.page_mid_ready = [False, False]
        self.page_sink_ready = [False, False]
        self.page_win_ready = [False, False]
        self.page_ids = None
        self.page_ids_bsz = 0
        self._head_idx = None

    def _maybe_expand_cpu(self, needed_tokens: int):
        if needed_tokens <= self.cpu_capacity:
            return
        if self.recall_stream is not None:
            self.recall_stream.synchronize()
        old_cap = self.cpu_capacity
        new_cap = old_cap
        while new_cap < needed_tokens:
            new_cap *= 2
        new_buf = torch.empty(
            (self.batch_size, new_cap, 2, self.n_kv_heads, self.head_dim),
            dtype=self.cpu_kv.dtype,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        if self.cpu_len > 0:
            new_buf[:, : self.cpu_len].copy_(self.cpu_kv[:, : self.cpu_len], non_blocking=False)
        self.cpu_kv = new_buf
        self.cpu_capacity = new_cap

    def copy_prefill_to_cpu(self, k: torch.Tensor, v: torch.Tensor):
        # k/v: [bsz, q_len, n_kv_heads, head_dim]
        q_len = k.shape[1]
        self._maybe_expand_cpu(q_len)
        self.cpu_kv[:, :q_len, 0].copy_(k, non_blocking=True)
        self.cpu_kv[:, :q_len, 1].copy_(v, non_blocking=True)
        self.cpu_len = q_len

        self.sink_len = min(self.sink_tokens, q_len)
        self.win_len = min(self.win_tokens, q_len)
        self.win_ptr = 0

        if self.sink_tokens > 0:
            self.sink_k = torch.empty(
                (self.batch_size, self.n_kv_heads, self.sink_tokens, self.head_dim),
                dtype=k.dtype,
                device=k.device,
            )
            self.sink_v = torch.empty_like(self.sink_k)
            if self.sink_len > 0:
                self.sink_k[:, :, : self.sink_len].copy_(
                    k[:, : self.sink_len].transpose(1, 2), non_blocking=True
                )
                self.sink_v[:, :, : self.sink_len].copy_(
                    v[:, : self.sink_len].transpose(1, 2), non_blocking=True
                )
        else:
            self.sink_k = None
            self.sink_v = None

        if self.win_tokens > 0:
            self.win_k = torch.empty(
                (self.batch_size, self.n_kv_heads, self.win_tokens, self.head_dim),
                dtype=k.dtype,
                device=k.device,
            )
            self.win_v = torch.empty_like(self.win_k)
            if self.win_len > 0:
                self.win_k[:, :, : self.win_len].copy_(
                    k[:, q_len - self.win_len : q_len].transpose(1, 2), non_blocking=True
                )
                self.win_v[:, :, : self.win_len].copy_(
                    v[:, q_len - self.win_len : q_len].transpose(1, 2), non_blocking=True
                )
        else:
            self.win_k = None
            self.win_v = None

        self._sync_sink_to_local_buffers()
        self._sync_sink_to_page_buffers()
        self._sync_window_to_page_buffers_full()

    def append_decode_token_to_cpu(
        self,
        k_tok: torch.Tensor,
        v_tok: torch.Tensor,
        src_stream: Optional[torch.cuda.Stream] = None,
    ):
        # k_tok/v_tok: [bsz, 1, n_kv_heads, head_dim]
        needed = self.cpu_len + 1
        self._maybe_expand_cpu(needed)
        if src_stream is None:
            src_stream = torch.cuda.current_stream(self.device)
        self.append_event.record(src_stream)
        with torch.cuda.stream(self.recall_stream):
            self.recall_stream.wait_event(self.append_event)
            self.cpu_kv[:, self.cpu_len : self.cpu_len + 1, 0].copy_(k_tok, non_blocking=True)
            self.cpu_kv[:, self.cpu_len : self.cpu_len + 1, 1].copy_(v_tok, non_blocking=True)
        self.cpu_len = needed
        k_tok_h = k_tok.transpose(1, 2)
        v_tok_h = v_tok.transpose(1, 2)

        if self.sink_tokens > 0:
            if self.sink_len < self.sink_tokens:
                self.sink_k[:, :, self.sink_len : self.sink_len + 1].copy_(k_tok_h, non_blocking=True)
                self.sink_v[:, :, self.sink_len : self.sink_len + 1].copy_(v_tok_h, non_blocking=True)
                self.sink_len += 1
                self._sync_sink_to_local_buffers()
                self._sync_sink_to_page_buffers()

        if self.win_tokens > 0:
            if self.win_len < self.win_tokens:
                self.win_k[:, :, self.win_len : self.win_len + 1].copy_(k_tok_h, non_blocking=True)
                self.win_v[:, :, self.win_len : self.win_len + 1].copy_(v_tok_h, non_blocking=True)
                self.win_len += 1
                self._sync_window_to_page_buffers_full()
            else:
                write_slot = self.win_ptr
                self.win_k[:, :, write_slot : write_slot + 1].copy_(k_tok_h, non_blocking=True)
                self.win_v[:, :, write_slot : write_slot + 1].copy_(v_tok_h, non_blocking=True)
                self._update_window_token_in_page_buffers(write_slot, k_tok_h, v_tok_h)
                self.win_ptr = (self.win_ptr + 1) % self.win_tokens

    def middle_bounds(self, seq_len: int) -> Optional[Tuple[int, int]]:
        lower = self.sink_tokens
        upper = seq_len - self.win_tokens
        if upper <= lower:
            return None
        if (upper - lower) < self.mid_tokens:
            return None
        return lower, upper

    def _sync_sink_to_local_buffers(self):
        if self.sink_len <= 0:
            self.local_sink_len_cached = [0, 0]
            return
        for i in (0, 1):
            lk = self.local_k_buf[i]
            lv = self.local_v_buf[i]
            if lk is None or lv is None:
                continue
            if self.local_sink_len_cached[i] == self.sink_len:
                continue
            lk[:, :, : self.sink_len].copy_(
                self.sink_k[: self.eff_batch, :, : self.sink_len], non_blocking=True
            )
            lv[:, :, : self.sink_len].copy_(
                self.sink_v[: self.eff_batch, :, : self.sink_len], non_blocking=True
            )
            self.local_sink_len_cached[i] = self.sink_len

    def _sync_sink_to_page_buffers(self):
        if self.sink_len < self.sink_tokens:
            return
        if self.n_sink_pages <= 0:
            return
        sink_k_pages = self.sink_k[: self.eff_batch].transpose(1, 2).reshape(
            self.eff_batch, self.n_sink_pages, self.page_size, self.n_kv_heads, self.head_dim
        )
        sink_v_pages = self.sink_v[: self.eff_batch].transpose(1, 2).reshape(
            self.eff_batch, self.n_sink_pages, self.page_size, self.n_kv_heads, self.head_dim
        )
        for i in (0, 1):
            if self.page_sink_ready[i]:
                continue
            pbuf = self.page_kv_buf[i]
            if pbuf is None:
                continue
            pbuf[:, : self.n_sink_pages, 0].copy_(sink_k_pages, non_blocking=True)
            pbuf[:, : self.n_sink_pages, 1].copy_(sink_v_pages, non_blocking=True)
            self.page_sink_ready[i] = True

    def _sync_window_to_page_buffers_full(self):
        if self.win_len < self.win_tokens or self.n_win_pages <= 0:
            return
        off = self.n_sink_pages + self.mid_pages
        win_pages_k = self.win_k[: self.eff_batch].transpose(1, 2).reshape(
            self.eff_batch, self.n_win_pages, self.page_size, self.n_kv_heads, self.head_dim
        )
        win_pages_v = self.win_v[: self.eff_batch].transpose(1, 2).reshape(
            self.eff_batch, self.n_win_pages, self.page_size, self.n_kv_heads, self.head_dim
        )
        for i in (0, 1):
            if self.page_win_ready[i]:
                continue
            pbuf = self.page_kv_buf[i]
            if pbuf is None:
                continue
            pbuf[:, off : off + self.n_win_pages, 0].copy_(win_pages_k, non_blocking=True)
            pbuf[:, off : off + self.n_win_pages, 1].copy_(win_pages_v, non_blocking=True)
            self.page_win_ready[i] = True

    def _update_window_token_in_page_buffers(
        self, slot: int, k_tok_h: torch.Tensor, v_tok_h: torch.Tensor
    ):
        if self.n_win_pages <= 0:
            return
        page_idx = int(slot // self.page_size)
        tok_off = int(slot % self.page_size)
        off = self.n_sink_pages + self.mid_pages + page_idx
        k_src = k_tok_h[: self.eff_batch, :, 0, :]
        v_src = v_tok_h[: self.eff_batch, :, 0, :]
        for i in (0, 1):
            if not self.page_win_ready[i]:
                continue
            pbuf = self.page_kv_buf[i]
            if pbuf is None:
                continue
            pbuf[:, off, 0, tok_off].copy_(k_src, non_blocking=True)
            pbuf[:, off, 1, tok_off].copy_(v_src, non_blocking=True)

    def _ensure_page_ids(self, bsz: int):
        if self.local_pages <= 0:
            self.page_ids = None
            self.page_ids_bsz = bsz
            return
        if self.page_ids is not None and self.page_ids_bsz == bsz:
            return
        base = torch.arange(self.local_pages, dtype=torch.int32, device=self.device)
        if self.eff_batch == 1 and bsz > 1:
            self.page_ids = base.view(1, -1).expand(bsz, -1).contiguous()
        else:
            offs = (
                torch.arange(bsz, dtype=torch.int32, device=self.device).view(-1, 1)
                * int(self.local_pages)
            )
            self.page_ids = offs + base.view(1, -1)
        self.page_ids_bsz = bsz

    def _copy_window_pages_to_page_buffer(self, dst_pages: torch.Tensor):
        win_len = self.win_len
        if win_len < self.win_tokens or self.n_win_pages <= 0:
            return
        # For q_len=1 with non-causal attention, key/value permutation is output-invariant.
        # Keep ring order and avoid per-step reordering/copying.
        self._sync_window_to_page_buffers_full()

    def _get_head_index(self, device: torch.device):
        if self.anchor_head_sample <= 0 or self.anchor_head_sample >= self.n_kv_heads:
            return None
        if self._head_idx is not None and self._head_idx.device == device:
            return self._head_idx
        k = int(max(1, min(self.anchor_head_sample, self.n_kv_heads)))
        if k == 1:
            idx = torch.zeros((1,), dtype=torch.long, device=device)
        else:
            idx = torch.floor(
                torch.arange(k, device=device, dtype=torch.float32)
                * (self.n_kv_heads / float(k))
            ).to(torch.long)
            idx = torch.clamp(idx, max=self.n_kv_heads - 1)
        self._head_idx = idx.contiguous()
        return self._head_idx

    def _expand_seed_anchors(self, anchors: torch.Tensor) -> torch.Tensor:
        # anchors: [eff_bsz, seed_k]
        if anchors.shape[1] == self.mid_pages:
            return anchors
        if anchors.shape[1] == 0:
            return torch.zeros(
                (anchors.shape[0], self.mid_pages),
                dtype=anchors.dtype,
                device=anchors.device,
            )
        idx = torch.linspace(
            0,
            anchors.shape[1] - 1,
            steps=self.mid_pages,
            device=anchors.device,
        ).round().to(torch.long)
        idx = idx.view(1, -1).expand(anchors.shape[0], -1)
        return anchors.gather(dim=1, index=idx)

    def init_anchors_from_prefill(self, q_last: torch.Tensor, k_full: torch.Tensor):
        # q_last: [bsz, 1, n_qo_heads, head_dim]
        # k_full: [bsz, seq_len, n_kv_heads, head_dim]
        if self.mid_pages <= 0:
            self.anchors = None
            return
        bounds = self.middle_bounds(k_full.shape[1])
        if bounds is None:
            self.anchors = None
            return
        lower, upper = bounds
        eff_q = q_last[: self.eff_batch]
        eff_k = k_full[: self.eff_batch]

        grouped_q = eff_q.view(
            self.eff_batch, self.n_kv_heads, self.n_q_per_kv, self.head_dim
        ).mean(dim=2)  # [eff, n_kv_heads, head_dim]
        k_hsd = eff_k.transpose(1, 2).contiguous()  # [eff, n_kv_heads, seq_len, head_dim]
        head_idx = self._get_head_index(grouped_q.device)
        if head_idx is not None:
            grouped_q = grouped_q.index_select(1, head_idx)
            k_hsd = k_hsd.index_select(1, head_idx)
        scores = torch.matmul(
            grouped_q.unsqueeze(2), k_hsd.transpose(2, 3)
        ).squeeze(2) * self._inv_sqrt_head_dim  # [eff, sampled_heads, seq_len]
        mid_scores = scores[..., lower:upper]  # [eff, sampled_heads, mid_len]
        k = min(self.seed_anchors, mid_scores.shape[-1])
        seed = torch.topk(mid_scores, k=k, dim=-1).indices + lower
        seed = seed.to(mid_scores.dtype).mean(dim=1).round().to(torch.int32)  # [eff, k], shared over kv heads
        seed, _ = torch.sort(seed, dim=-1)
        anchors = self._expand_seed_anchors(seed)
        anchors, _ = torch.sort(anchors, dim=-1)
        self.anchors = anchors.to(device=torch.device("cpu"), dtype=torch.int32).contiguous()

    def _tile_starts(self, anchors: torch.Tensor, lower: int, upper: int) -> torch.Tensor:
        # anchors: [eff_bsz, mid_pages]
        max_start = upper - self.page_size
        if max_start < lower:
            return torch.full_like(anchors, lower)

        ideal = torch.clamp(anchors - self.page_size // 2, min=lower, max=max_start)
        ideal, _ = torch.sort(ideal, dim=-1)

        penalty = torch.arange(self.mid_pages, device=anchors.device, dtype=ideal.dtype)
        penalty = penalty * self.page_size
        y = ideal - penalty.view(1, -1)
        starts = torch.cummax(y, dim=-1).values + penalty.view(1, -1)

        upper_bounds = max_start - torch.arange(
            self.mid_pages - 1, -1, -1, device=anchors.device, dtype=ideal.dtype
        ) * self.page_size
        starts = torch.minimum(starts, upper_bounds.view(1, -1))

        y2 = starts - penalty.view(1, -1)
        starts = torch.cummax(y2, dim=-1).values + penalty.view(1, -1)
        starts = torch.clamp(starts, min=lower, max=max_start)
        return starts.to(dtype=torch.int32).contiguous()

    def build_starts(self, seq_len: int) -> Optional[torch.Tensor]:
        if self.mid_pages <= 0 or self.anchors is None:
            return None
        bounds = self.middle_bounds(seq_len)
        if bounds is None:
            return None
        lower, upper = bounds
        return self._tile_starts(self.anchors, lower, upper)

    def _fallback_recall(self, starts: torch.Tensor, out_buf: torch.Tensor):
        # starts: [eff_bsz, mid_pages], out_buf: [eff_bsz, mid_tokens, 2, n_kv_heads, head_dim]
        starts_cpu = starts.to(device=torch.device("cpu"), dtype=torch.long)
        for b in range(starts_cpu.shape[0]):
            for p in range(self.mid_pages):
                s = int(starts_cpu[b, p].item())
                e = s + self.page_size
                out_buf[b, p * self.page_size : (p + 1) * self.page_size].copy_(
                    self.cpu_kv[b, s:e],
                    non_blocking=True,
                )

    def _delta_recall_from_active(
        self, starts_cpu_i32: torch.Tensor, out_buf: torch.Tensor, in_place: bool = False
    ) -> bool:
        # Reuse previous active middle buffer and patch only changed pages.
        if self.active_starts is None:
            return False
        prev = self.active_starts
        if prev.device.type != "cpu":
            return False
        if prev.shape != starts_cpu_i32.shape:
            return False
        changed = starts_cpu_i32.ne(prev)
        changed_cnt = int(changed.sum().item())
        if changed_cnt <= 0:
            if not in_place:
                out_buf.copy_(self.gpu_mid[self.active_idx], non_blocking=True)
            return True
        if changed_cnt > int(self.delta_max_pages):
            return False

        if not in_place:
            out_buf.copy_(self.gpu_mid[self.active_idx], non_blocking=True)
        for b in range(changed.shape[0]):
            idxs = torch.nonzero(changed[b], as_tuple=False).flatten()
            for p_t in idxs:
                p = int(p_t.item())
                s = int(starts_cpu_i32[b, p].item())
                e = s + self.page_size
                out_buf[b, p * self.page_size : (p + 1) * self.page_size].copy_(
                    self.cpu_kv[b, s:e],
                    non_blocking=True,
                )
        return True

    def launch_prefetch(self, starts: torch.Tensor, target_seq_len: int):
        # starts: [eff_bsz, mid_pages]
        if self.mid_pages <= 0 or starts is None:
            self.prefetch_ready = False
            return
        if starts.dtype != torch.int32 or (not starts.is_contiguous()):
            starts_i32 = starts.to(dtype=torch.int32).contiguous()
        else:
            starts_i32 = starts
        starts_cpu = starts_i32 if starts_i32.device.type == "cpu" else starts_i32.to("cpu")
        pending_idx = 1 - self.active_idx
        out_buf = self.gpu_mid[pending_idx]

        with torch.cuda.stream(self.recall_stream):
            used_delta = False
            if self.active_starts is not None:
                # In-place delta patch on active buffer avoids large D2D copy.
                used_delta = self._delta_recall_from_active(
                    starts_cpu, self.gpu_mid[self.active_idx], in_place=True
                )
                if used_delta:
                    pending_idx = self.active_idx
                    out_buf = self.gpu_mid[pending_idx]
            used_cuda = False
            if (not used_delta) and self.use_cuda_token_recall and hasattr(kernels, "recall_tokens_linear"):
                try:
                    kernels.recall_tokens_linear(
                        starts_i32,
                        self.cpu_kv[: self.eff_batch],
                        out_buf,
                        self.cpu_len,
                    )
                    used_cuda = True
                except Exception:
                    used_cuda = False
            if (not used_delta) and (not used_cuda):
                self._fallback_recall(starts_cpu, out_buf)
            self.pending_idx = pending_idx
            self.local_mid_ready[pending_idx] = False
            self.page_mid_ready[pending_idx] = False
            if self.local_k_buf[self.pending_idx] is not None:
                local_k = self.local_k_buf[self.pending_idx]
                local_v = self.local_v_buf[self.pending_idx]
                off = self.sink_len
                mid_k_hsd = out_buf[:, :, 0].permute(0, 2, 1, 3)
                mid_v_hsd = out_buf[:, :, 1].permute(0, 2, 1, 3)
                local_k[:, :, off : off + self.mid_tokens].copy_(mid_k_hsd, non_blocking=True)
                local_v[:, :, off : off + self.mid_tokens].copy_(mid_v_hsd, non_blocking=True)
                self.local_mid_ready[self.pending_idx] = True
            if self.page_kv_buf[self.pending_idx] is not None:
                self._sync_sink_to_page_buffers()
                pbuf = self.page_kv_buf[self.pending_idx]
                mid_pages_k = out_buf[:, :, 0].view(
                    self.eff_batch,
                    self.mid_pages,
                    self.page_size,
                    self.n_kv_heads,
                    self.head_dim,
                )
                mid_pages_v = out_buf[:, :, 1].view(
                    self.eff_batch,
                    self.mid_pages,
                    self.page_size,
                    self.n_kv_heads,
                    self.head_dim,
                )
                offp = self.n_sink_pages
                pbuf[:, offp : offp + self.mid_pages, 0].copy_(mid_pages_k, non_blocking=True)
                pbuf[:, offp : offp + self.mid_pages, 1].copy_(mid_pages_v, non_blocking=True)
                self.page_mid_ready[self.pending_idx] = True
            self.prefetch_event.record(self.recall_stream)

        self.pending_seq_len = int(target_seq_len)
        self.pending_starts = starts
        self.prefetch_ready = True

    def activate_prefetch(self, seq_len: int, compute_stream: torch.cuda.Stream):
        if (not self.prefetch_ready) or self.pending_seq_len != int(seq_len):
            return False
        compute_stream.wait_event(self.prefetch_event)
        self.active_idx = self.pending_idx
        self.active_seq_len = self.pending_seq_len
        self.active_starts = self.pending_starts
        return True

    def paged_kv(self, bsz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_page_ids(bsz)
        pbuf = self.page_kv_buf[self.active_idx]
        if pbuf is None:
            raise RuntimeError("Echo page buffer is not initialized")
        if not self.page_mid_ready[self.active_idx]:
            mid = self.gpu_mid[self.active_idx]
            mid_pages_k = mid[:, :, 0].view(
                self.eff_batch, self.mid_pages, self.page_size, self.n_kv_heads, self.head_dim
            )
            mid_pages_v = mid[:, :, 1].view(
                self.eff_batch, self.mid_pages, self.page_size, self.n_kv_heads, self.head_dim
            )
            offp = self.n_sink_pages
            pbuf[:, offp : offp + self.mid_pages, 0].copy_(mid_pages_k, non_blocking=True)
            pbuf[:, offp : offp + self.mid_pages, 1].copy_(mid_pages_v, non_blocking=True)
            self.page_mid_ready[self.active_idx] = True
        self._sync_sink_to_page_buffers()
        self._copy_window_pages_to_page_buffer(pbuf)

        if self.eff_batch == 1 and bsz > 1:
            kv_data = pbuf[:1].reshape(
                self.local_pages, 2, self.page_size, self.n_kv_heads, self.head_dim
            )
        else:
            kv_data = pbuf.reshape(
                self.eff_batch * self.local_pages,
                2,
                self.page_size,
                self.n_kv_heads,
                self.head_dim,
            )
        return kv_data, self.page_ids

    def current_middle_kv(self, bsz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mid = self.gpu_mid[self.active_idx]  # [eff_bsz, mid_tokens, 2, n_kv_heads, head_dim]
        if self.eff_batch == 1 and bsz > 1:
            mid = mid[:1]
        k_mid = mid[:, :, 0].permute(0, 2, 1, 3).contiguous()  # [eff, n_kv_heads, mid_tokens, head_dim]
        v_mid = mid[:, :, 1].permute(0, 2, 1, 3).contiguous()
        return k_mid, v_mid

    def _copy_window_ordered(self, dst_k: torch.Tensor, dst_v: torch.Tensor, offset: int):
        win_len = self.win_len
        if win_len <= 0:
            return
        # Permutation of keys/values does not change attention output for single-query non-causal decode.
        dst_k[:, :, offset : offset + win_len].copy_(
            self.win_k[: self.eff_batch, :, :win_len], non_blocking=True
        )
        dst_v[:, :, offset : offset + win_len].copy_(
            self.win_v[: self.eff_batch, :, :win_len], non_blocking=True
        )

    def local_kv(self, bsz: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sink_len = self.sink_len
        win_len = self.win_len
        total = sink_len + self.mid_tokens + win_len
        local_k = self.local_k_buf[self.active_idx][:, :, :total]
        local_v = self.local_v_buf[self.active_idx][:, :, :total]
        if not self.local_mid_ready[self.active_idx]:
            k_mid, v_mid = self.current_middle_kv(bsz)
            local_k[:, :, sink_len : sink_len + self.mid_tokens].copy_(k_mid, non_blocking=True)
            local_v[:, :, sink_len : sink_len + self.mid_tokens].copy_(v_mid, non_blocking=True)
            self.local_mid_ready[self.active_idx] = True
        if win_len > 0:
            self._copy_window_ordered(local_k, local_v, sink_len + self.mid_tokens)
        k_mid = local_k[:, :, sink_len : sink_len + self.mid_tokens]
        if self.eff_batch == 1 and bsz > 1:
            return local_k[:1], local_v[:1], k_mid
        return local_k, local_v, k_mid

    def update_anchors_from_middle(self, q: torch.Tensor, k_mid: torch.Tensor):
        # q: [bsz, 1, n_qo_heads, head_dim]
        if self.mid_pages <= 0:
            return
        if self.active_starts is None:
            return
        eff_q = q[: self.eff_batch]
        eff_k_mid = k_mid[: self.eff_batch]
        starts = self.active_starts

        grouped_q = eff_q.view(
            self.eff_batch, self.n_kv_heads, self.n_q_per_kv, self.head_dim
        ).mean(dim=2)  # [eff, n_kv_heads, head_dim]
        head_idx = self._get_head_index(grouped_q.device)
        if head_idx is not None:
            grouped_q = grouped_q.index_select(1, head_idx)
            eff_k_mid = eff_k_mid.index_select(1, head_idx)
        scores = torch.matmul(
            grouped_q.unsqueeze(2), eff_k_mid.transpose(2, 3)
        ).squeeze(2) * self._inv_sqrt_head_dim  # [eff, sampled_heads, mid_tokens]
        page_probs = scores.view(
            self.eff_batch, scores.shape[1], self.mid_pages, self.page_size
        )
        local_best = page_probs.argmax(dim=-1).float().mean(dim=1).round().to(torch.int32)
        anchors = starts + local_best
        anchors, _ = torch.sort(anchors, dim=-1)
        self.anchors = anchors.to(device=torch.device("cpu"), dtype=torch.int32).contiguous()

    def update_anchors_from_active_mid(self, q: torch.Tensor):
        # Use active gpu_mid directly to avoid extra permute/contiguous on flashinfer path.
        if self.mid_pages <= 0:
            return
        if self.active_starts is None:
            return
        eff_q = q[: self.eff_batch]
        mid = self.gpu_mid[self.active_idx][: self.eff_batch, :, 0]  # [eff, mid_tokens, n_kv_heads, head_dim]
        grouped_q = eff_q.view(
            self.eff_batch, self.n_kv_heads, self.n_q_per_kv, self.head_dim
        ).mean(dim=2)  # [eff, n_kv_heads, head_dim]
        head_idx = self._get_head_index(grouped_q.device)
        if head_idx is not None:
            grouped_q = grouped_q.index_select(1, head_idx)
            mid = mid.index_select(2, head_idx)
        # [eff, sampled_heads, mid_tokens]
        scores = torch.einsum("ehd,ethd->eht", grouped_q, mid) * self._inv_sqrt_head_dim
        page_probs = scores.view(
            self.eff_batch, scores.shape[1], self.mid_pages, self.page_size
        )
        local_best = page_probs.argmax(dim=-1).float().mean(dim=1).round().to(torch.int32)
        local_best_cpu = local_best.to(device=torch.device("cpu"), dtype=torch.int32)
        anchors = self.active_starts + local_best_cpu
        anchors, _ = torch.sort(anchors, dim=-1)
        self.anchors = anchors.to(dtype=torch.int32).contiguous()

    def attend(self, q: torch.Tensor, local_k: torch.Tensor, local_v: torch.Tensor) -> torch.Tensor:
        # q: [bsz, 1, n_qo_heads, head_dim]
        # local_k/local_v: [bsz, n_kv_heads, s, head_dim]
        bsz = q.shape[0]
        if local_k.shape[0] != bsz:
            q_use = q[:1]
        else:
            q_use = q
        qh = q_use.transpose(1, 2).contiguous()        # [bsz, n_qo_heads, 1, head_dim]
        kh = local_k
        vh = local_v

        try:
            out = F.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
            )
        except TypeError:
            kh = kh.repeat_interleave(self.n_q_per_kv, dim=1)
            vh = vh.repeat_interleave(self.n_q_per_kv, dim=1)
            out = F.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        out = out.transpose(1, 2).contiguous()  # [bsz, 1, n_qo_heads, head_dim]
        if out.shape[0] == 1 and bsz > 1:
            out = out.expand(bsz, -1, -1, -1).contiguous()
        return out
