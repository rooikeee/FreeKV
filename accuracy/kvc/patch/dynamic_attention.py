from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import repeat_kv
import math


def repeat_kv_BLH(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None,:].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _ensure_quest_timing_state(attn_module) -> None:
    if not hasattr(attn_module, "_quest_token_select_events"):
        attn_module._quest_token_select_events = []
    if not hasattr(attn_module, "_quest_attn_compute_events"):
        attn_module._quest_attn_compute_events = []


def _quest_timing_start(attn_module):
    if not torch.cuda.is_available():
        return None
    if not hasattr(attn_module, "q_proj") or (not attn_module.q_proj.weight.is_cuda):
        return None
    _ensure_quest_timing_state(attn_module)
    start_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    return start_evt


def _quest_timing_stop(attn_module, stage: str, start_evt):
    if start_evt is None:
        return
    end_evt = torch.cuda.Event(enable_timing=True)
    end_evt.record()
    _ensure_quest_timing_state(attn_module)
    if stage == "select":
        attn_module._quest_token_select_events.append((start_evt, end_evt))
    elif stage == "attn":
        attn_module._quest_attn_compute_events.append((start_evt, end_evt))


def quest_sel(q, min_k_for_sel, max_k_for_sel, GQA_policy, num_heads, num_kv_heads):
    if GQA_policy in ["maxQ", "avgQ"]:
        # [bsz, 1, num_kv_heads, head_dim]
        if GQA_policy == "maxQ":
            grouped_q = q.reshape(
                q.shape[0], q.shape[1], num_kv_heads, num_heads // num_kv_heads, q.shape[-1]
            ).max(dim=-2).values
        elif GQA_policy == "avgQ":
            grouped_q = q.reshape(
                q.shape[0], q.shape[1], num_kv_heads, num_heads // num_kv_heads, q.shape[-1]
            ).mean(dim=-2)
        # [bsz, num_used_pages, num_kv_heads, head_dim]
        q_min_k = grouped_q * min_k_for_sel
        q_max_k = grouped_q * max_k_for_sel
        # [bsz, num_kv_heads, num_used_pages]
        max_qk = torch.maximum(q_min_k, q_max_k).sum(dim=-1).transpose(1, 2)
    elif GQA_policy in ["maxS", "avgS", "maxSM", "avgSM", "avgSdM"]:
        # [bsz, num_used_pages, num_heads, head_dim]
        q_min_k = q * repeat_kv_BLH(min_k_for_sel, num_heads // num_kv_heads)
        q_max_k = q * repeat_kv_BLH(max_k_for_sel, num_heads // num_kv_heads)
        # [bsz, num_heads, num_used_pages]
        max_qk = torch.maximum(q_min_k, q_max_k).sum(dim=-1).transpose(1, 2)
        # [bsz, num_kv_heads, num_used_pages]
        if GQA_policy == "maxS":
            max_qk = max_qk.reshape(
                max_qk.shape[0], num_kv_heads, num_heads//num_kv_heads, max_qk.shape[-1]
            ).max(dim=-2).values
        elif GQA_policy == "avgS":
            max_qk = max_qk.reshape(
                max_qk.shape[0], num_kv_heads, num_heads//num_kv_heads, max_qk.shape[-1]
            ).mean(dim=-2)
        elif GQA_policy == "maxSM":
            max_qk = F.softmax(max_qk, dim=-1)
            max_qk = max_qk.reshape(
                max_qk.shape[0], num_kv_heads, num_heads//num_kv_heads, max_qk.shape[-1]
            ).max(dim=-2).values
        elif GQA_policy == "avgSM":
            max_qk = F.softmax(max_qk, dim=-1)
            max_qk = max_qk.reshape(
                max_qk.shape[0], num_kv_heads, num_heads//num_kv_heads, max_qk.shape[-1]
            ).mean(dim=-2)
        elif GQA_policy == "avgSdM":
            max_qk = F.softmax(max_qk / math.sqrt(q.shape[-1]), dim=-1)
            max_qk = max_qk.reshape(
                max_qk.shape[0], num_kv_heads, num_heads//num_kv_heads, max_qk.shape[-1]
            ).mean(dim=-2)
    else:
        assert False
    
    return max_qk


def quest_arkv_attn(self,
              q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor, 
              padding_mask: Optional[torch.LongTensor] = None,
              update_only=False) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, seqlen=1, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    num_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    head_dim = q.shape[-1]
    sink_size = self.sink_size
    recent_size = self.recent_size
    page_size = self.page_size
    token_budget = self.token_budget
    ll_token_budget = self.ll_token_budget

    page_budget = token_budget // page_size
    ll_page_budget = ll_token_budget // page_size

    kv_seq_len = k.shape[1]
    if self.num_pages == 0:
        # first decoding step that exceeds sink, init pages
        num_init_pages = (kv_seq_len - sink_size) // page_size
        assert num_init_pages < self.max_k.shape[1]
        if num_init_pages > 0:  # maybe sink is large
            paged_k = k[:, sink_size: sink_size + num_init_pages*page_size, :, :].reshape(
                k.shape[0], num_init_pages, page_size, num_kv_heads, head_dim
            )
            mins = paged_k.min(dim=2).values
            maxs = paged_k.max(dim=2).values
            if self.page_rep == "arkv":
                centers = (maxs + mins) / 2
                dists = (
                    (
                        centers.reshape(*paged_k.shape[:2], 1, -1, self.head_dim)
                        - paged_k
                    ).abs().mean(dim=2)
                )
                self.min_k[:, :num_init_pages, ...] = centers - dists
                self.max_k[:, :num_init_pages, ...] = centers + dists
            else:
                self.min_k[:, :num_init_pages, ...] = mins
                self.max_k[:, :num_init_pages, ...] = maxs
            self.num_pages += num_init_pages
    elif (kv_seq_len - sink_size) // page_size > self.num_pages:
        # new page comes
        new_paged_k = k[:, sink_size + self.num_pages*page_size:, :, :].reshape(
            k.shape[0], page_size, num_kv_heads, head_dim
        )
        mins = new_paged_k.amin(dim=1)
        maxs = new_paged_k.amax(dim=1)
        if self.page_rep == "arkv":
            centers = (maxs + mins) / 2
            dists = (
                (
                    centers.reshape(new_paged_k.shape[0], 1, -1, self.head_dim)
                    - new_paged_k
                ).abs().mean(dim=1)
            )
            self.min_k[:, self.num_pages, ...] = centers - dists
            self.max_k[:, self.num_pages, ...] = centers + dists
        else:
            self.min_k[:, self.num_pages, ...] = mins
            self.max_k[:, self.num_pages, ...] = maxs
        self.num_pages += 1

    if self.num_pages >= self.max_k.shape[1]:
        new_min_k = torch.empty((1, self.alloc_len, self.min_k.shape[-2], self.min_k.shape[-1]), 
                                device=self.min_k.device, dtype=self.min_k.dtype)
        self.min_k = torch.cat([self.min_k, new_min_k], dim=1)
        new_max_k = torch.empty((1, self.alloc_len, self.max_k.shape[-2], self.max_k.shape[-1]), 
                                device=self.max_k.device, dtype=self.max_k.dtype)
        self.max_k = torch.cat([self.max_k, new_max_k], dim=1)
    
    if update_only:
        return None
    
    use_spec_ret_steps = hasattr(self, "spec_ret_steps")
    if use_spec_ret_steps:
        self.q_cache[:, self.q_ptr, ...] = q.squeeze(1)
        self.q_ptr = (self.q_ptr + 1) % self.spec_ret_steps

    do_compress = (kv_seq_len > sink_size + recent_size + max(token_budget, ll_token_budget))
    if token_budget < 0 or not do_compress:
        attn_t0 = _quest_timing_start(self)
        out = self._flash_attention_forward(
            q, k, v, padding_mask, q.shape[1], dropout=0.0
        )
        _quest_timing_stop(self, "attn", attn_t0)
        return out

    sel_t0 = _quest_timing_start(self)
    last_page_size = (kv_seq_len - sink_size) % page_size
    # to achieve an averaged fix recent size
    num_recent_page = recent_size // page_size - (last_page_size > page_size//2)
    # recent pages + last page
    recent_size = num_recent_page * page_size + last_page_size

    # exclude the recent pages for selection
    # [1, num_used_pages, num_kv_heads, head_dim]
    min_k_for_sel = self.min_k[:, :self.num_pages - num_recent_page, ...]
    max_k_for_sel = self.max_k[:, :self.num_pages - num_recent_page, ...]

    # selection with current or past query
    if token_budget > 0:
        if not use_spec_ret_steps:
            probe_q = q
        else:
            past_q = self.q_cache[:, self.q_ptr, ...].unsqueeze(1)
            if hasattr(self, "correct_sim") and self.correct_sim is not None:
                # [1, 1, num_heads, head_dim] => [1, 1, num_heads]
                sim = F.cosine_similarity(q, past_q, dim=-1) 
                # [1, 1, num_heads] => [1, 1, num_kv_heads]
                if self.corr_group == "max":
                    kv_head_sim = sim.reshape(1, 1, num_kv_heads, -1).max(dim=-1).values
                elif self.corr_group == "avg":
                    kv_head_sim = sim.reshape(1, 1, num_kv_heads, -1).mean(dim=-1)
                kv_head_corr = kv_head_sim < self.correct_sim
                if torch.any(kv_head_corr):
                    # [1, 1, num_kv_heads] => [1, 1, num_heads]
                    corr = kv_head_corr.repeat_interleave(num_heads//num_kv_heads, dim=-1)
                    probe_q = torch.where(corr.unsqueeze(-1), q, past_q)
                    # stat
                    self.num_correct += 1
                    self.num_correct_kv_heads += kv_head_corr.sum()
                else:
                    probe_q = past_q
            else:
                probe_q = past_q

        max_qk = quest_sel(probe_q, min_k_for_sel, max_k_for_sel, self.GQA_policy, num_heads, num_kv_heads)
        # [bsz, num_kv_heads, page_budget]
        _, sel_page_indices = torch.topk(max_qk, page_budget, dim=-1)
    
    # selection with last layer query
    if self.layer_idx > 0 and ll_page_budget > 0:
        llq_ptr = (self.last_layer_attn.q_ptr - 1) % self.spec_ret_steps
        llq = self.last_layer_attn.q_cache[:, llq_ptr, ...]
        max_qk = quest_sel(llq, min_k_for_sel, max_k_for_sel, self.GQA_policy, num_heads, num_kv_heads)
        _, ll_sel_page_indices = torch.topk(max_qk, ll_page_budget, dim=-1)
        # [bsz, num_kv_heads, (up to) page_budget + ll_page_budget]
        # still maybe repeated pages...
        if token_budget == 0:
            sel_page_indices = ll_sel_page_indices
        else:
            sel_page_indices = torch.unique(torch.cat([sel_page_indices, ll_sel_page_indices], dim=-1), dim=-1)
    # [bsz, num_kv_heads, page_budget, page_size]
    sel_token_indices = (sink_size + sel_page_indices*page_size).unsqueeze(-1).expand(-1, -1, -1, page_size)
    sel_token_indices = sel_token_indices + torch.arange(
        page_size, device=sel_token_indices.device
    ).reshape(1, 1, 1, page_size)

    # [bsz, num_sel_tokens, num_kv_heads, head_dim]
    sel_token_indices = sel_token_indices.reshape(
        sel_token_indices.shape[0], sel_token_indices.shape[1], -1
    ).transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, head_dim)

    sel_k = torch.gather(k, dim=1, index=sel_token_indices)
    sel_v = torch.gather(v, dim=1, index=sel_token_indices)

    used_k = torch.cat([
        k[:, :sink_size, ...], sel_k, k[:, -recent_size:, ...],
    ], dim=1)
    used_v = torch.cat([
        v[:, :sink_size, ...], sel_v, v[:, -recent_size:, ...],
    ], dim=1)
    used_pmask = torch.cat([
        padding_mask[:, :sink_size], 
        self.budget_ones,
        padding_mask[:, -recent_size:],
    ], dim=1) if padding_mask is not None else None

    _quest_timing_stop(self, "select", sel_t0)
    attn_t0 = _quest_timing_start(self)
    out = self._flash_attention_forward(
        q, used_k, used_v, used_pmask, q.shape[1], dropout=0.0
    )
    _quest_timing_stop(self, "attn", attn_t0)
    return out


def raas_attn(self,
              q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              padding_mask: Optional[torch.LongTensor] = None,) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, seqlen=1, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    bsz, q_len, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    sink_size = self.sink_size
    recent_size = self.recent_size
    page_size = self.page_size
    token_budget = self.token_budget
    page_budget = self.page_budget

    if token_budget < 0:
        return self._flash_attention_forward(
            q, k, v, padding_mask, q.shape[1], dropout=0.0
        )

    kv_seq_len = self.kv_seq_len
    if self.num_pages == 0:
        # first decoding step that exceeds sink, init pages
        num_init_pages = (kv_seq_len - sink_size - recent_size) // page_size
        # assert num_init_pages < page_budget
        if num_init_pages > 0:  # maybe sink is large
            if num_init_pages > page_budget:
                num_init_pages = page_budget    # keep the start tokens
                probe_q = q[:, -1:, ...].transpose(1, 2).view(
                    bsz, num_kv_heads, num_heads//num_kv_heads, 1, head_dim
                ).mean(dim=2)
                # [bsz, num_kv_heads, 1, pf_len]
                pf_attn_weights = torch.matmul(probe_q, k.transpose(1,2).transpose(2,3))
                # [bsz, num_kv_heads, pf_len]
                _, sel_indices = torch.topk(pf_attn_weights[..., sink_size: -recent_size],
                                            num_init_pages*page_size, dim=-1)
                sel_indices = sink_size + sel_indices.squeeze(2)
                # gather k, v using sel_indices and store to self.cached_k and self.cached_v 
                # [bsz, pf_len, num_kv_heads]
                transposed_indices = sel_indices.transpose(1, 2)
                # [bsz, pf_len, num_kv_heads, head_dim]
                expanded_indices = transposed_indices.unsqueeze(-1).expand(
                    bsz, num_init_pages*page_size, num_kv_heads, head_dim
                )
                gathered_k = torch.gather(k, 1, expanded_indices)
                gathered_v = torch.gather(v, 1, expanded_indices)

                self.cached_k[:, :num_init_pages*page_size, :, :] = gathered_k
                self.cached_v[:, :num_init_pages*page_size, :, :] = gathered_v
            else:
                self.cached_k[:, :num_init_pages*page_size, ...] = k[:, sink_size: sink_size + num_init_pages*page_size, ...]
                self.cached_v[:, :num_init_pages*page_size, ...] = v[:, sink_size: sink_size + num_init_pages*page_size, ...]
            if page_size > 1:
                paged_k = k[:, sink_size: sink_size + num_init_pages*page_size, :, :].reshape(
                    k.shape[0], num_init_pages, page_size, num_kv_heads, head_dim
                )
                self.min_k[:, :num_init_pages, ...] = paged_k.amin(dim=2)
                self.max_k[:, :num_init_pages, ...] = paged_k.amax(dim=2)
            self.num_pages += num_init_pages
    elif (kv_seq_len - sink_size - recent_size) % page_size == 0:
        # new page comes
        k_to_fill = k[:, -recent_size - page_size: -recent_size, ...]
        v_to_fill = v[:, -recent_size - page_size: -recent_size, ...]
        if self.num_pages < page_budget:
            # not reaching budget
            page_id = self.num_pages
            self.cached_k[:, page_id*page_size: (page_id+1)*page_size, ...] = k_to_fill
            self.cached_v[:, page_id*page_size: (page_id+1)*page_size, ...] = v_to_fill
            if page_size > 1:
                self.min_k[:, page_id, ...] = k_to_fill.amin(dim=1)
                self.max_k[:, page_id, ...] = k_to_fill.amax(dim=1)
            self.num_pages += 1
        else:
            # look for the oldest timestamp
            # [bsz, 1, num_kv_heads]
            evict_page = torch.min(self.page_timestamp, dim=1).indices.unsqueeze(1)
            evict_tokens = evict_page * page_size
            # [bsz, 1, num_kv_heads] + [bsz, page_size, 1] => [bsz, page_size, num_kv_heads]
            evict_tokens = (evict_tokens + self.page_indices)
            # [num_kv_heads]
            self.cached_k[self.batch_ind, evict_tokens, self.head_indices, :] = k_to_fill
            self.cached_v[self.batch_ind, evict_tokens, self.head_indices, :] = v_to_fill
            if page_size > 1:
                self.min_k[self.batch_ind, evict_page, self.head_indices, :] = k_to_fill.amin(dim=1, keepdim=True)
                self.max_k[self.batch_ind, evict_page, self.head_indices, :] = k_to_fill.amax(dim=1, keepdim=True)

            # if self.layer_idx == 0:
            #     print(token_indices, token_indices.shape, head_indices)
    
    # for raas, compress when seq_len == sink+recent+budget, for updating timestamp
    do_compress = (kv_seq_len >= sink_size + recent_size + token_budget)
    if q.shape[1] == k.shape[1] or not do_compress:
        return self._flash_attention_forward(
            q, k, v, padding_mask, q.shape[1], dropout=0.0
        )

    used_k = torch.cat([
        k[:, :sink_size, ...], 
        self.cached_k[:, :self.num_pages*page_size, ...], 
        k[:, -recent_size:, ...],
    ], dim=1)
    used_v = torch.cat([
        v[:, :sink_size, ...], 
        self.cached_v[:, :self.num_pages*page_size, ...], 
        v[:, -recent_size:, ...],
    ], dim=1)
    used_pmask = torch.cat([
        padding_mask[:, :sink_size], 
        self.budget_ones,
        padding_mask[:, -recent_size:],
    ], dim=1) if padding_mask is not None else None
    
    # update timestamp
    q = q.transpose(1, 2)
    if page_size == 1:
        # compute attention scores
        used_k = used_k.transpose(1, 2)
        used_v = used_v.transpose(1, 2)
        used_k = repeat_kv(used_k, num_heads // num_kv_heads)
        used_v = repeat_kv(used_v, num_heads // num_kv_heads)
        # [bsz, num_heads, 1, used_len] 
        attn_weights = torch.matmul(q, used_k.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        # [bsz, num_kv_heads, 1, token_budget] 
        middle_score_avg = attn_weights[..., sink_size: -recent_size].reshape(
            bsz, num_kv_heads, num_heads//num_kv_heads, q_len, token_budget
        ).mean(dim=2)
        # [bsz, token_budget, num_kv_heads] 
        update_mask = (middle_score_avg > self.alpha).squeeze(-2).transpose(1, 2)
        self.page_timestamp[update_mask] = kv_seq_len

        # btw, give the output
        attn_output = torch.matmul(attn_weights, used_v)
        attn_output = attn_output.transpose(1, 2)
        return attn_output
    else:
        # quest-like updating for page_size > 1
        min_k = self.min_k.transpose(1, 2)
        max_k = self.max_k.transpose(1, 2)
        # [bsz, num_heads, page_budget, head_dim]
        min_k = repeat_kv(min_k, num_heads // num_kv_heads)
        max_k = repeat_kv(max_k, num_heads // num_kv_heads)
        q_min_k = q * min_k
        q_max_k = q * max_k
        # [bsz, num_heads, page_budget]
        page_weights = torch.maximum(q_min_k, q_max_k).sum(dim=-1)
        page_weights = nn.functional.softmax(page_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        # [bsz, num_kv_heads, page_budget]
        page_score_avg = page_weights.reshape(
            bsz, num_kv_heads, num_heads//num_kv_heads, page_budget
        ).mean(dim=2)
        # [bsz, page_budget, num_kv_heads]
        update_mask = (page_score_avg > self.alpha).transpose(1, 2)
        self.page_timestamp[update_mask] = kv_seq_len

        q = q.transpose(1, 2)
        return self._flash_attention_forward(
            q, used_k, used_v, used_pmask, q.shape[1], dropout=0.0
        )
