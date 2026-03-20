from typing import Optional, Tuple
from functools import wraps

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.cache_utils import Cache

# use these classes just for hint
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaRMSNorm,
)

from freekv.infer_state import InferState
from freekv import kernels
from flashinfer import apply_rope_inplace, apply_llama31_rope_inplace

import time
import os

FORCE_CORR_RATE = os.getenv("FORCE_CORR_RATE")
if FORCE_CORR_RATE is not None:
    rate = float(FORCE_CORR_RATE)
    MAX_BATCH, MAX_KV_HEAD = 16, 8
    rate_tensor = torch.full((MAX_BATCH, MAX_KV_HEAD), rate, device='cuda')
    _force_to_corr = torch.bernoulli(rate_tensor).bool()

NO_SPEC_RET_LAYER_SET = set()
no_spec_ret_layer_env = os.getenv("NO_SPEC_RET_LAYER")
if no_spec_ret_layer_env:
    NO_SPEC_RET_LAYER_SET = {
        int(x.strip()) for x in no_spec_ret_layer_env.split(",") if x.strip()
    }

ALWAYS_CORR_LAYER_SET = set()
always_corr_layer_env = os.getenv("ALWAYS_CORR_LAYER")
if always_corr_layer_env:
    ALWAYS_CORR_LAYER_SET = {
        int(x.strip()) for x in always_corr_layer_env.split(",") if x.strip()
    }

@torch.compile(mode="reduce-overhead", fullgraph=True)
def get_corr_torch_compile(
    query_states: torch.Tensor,
    last_step_q: torch.Tensor,
    n_kv_heads: int,
    corr: float,
):
    # [bsz, q_len, num_heads]
    sim = F.cosine_similarity(last_step_q, query_states, dim=-1)
    # [bsz, num_kv_heads]
    bsz = query_states.shape[0]
    sim = sim.view(bsz, n_kv_heads, -1).mean(dim=-1)
    to_corr = torch.lt(sim, corr)
    need_corr = torch.any(to_corr)
    return to_corr, need_corr

def _freekv_rms_norm_forward_streamed(self: LlamaRMSNorm, hidden_states, infer_state: InferState):
    with torch.cuda.stream(infer_state.compute_stream):
        output = kernels.rms_norm(hidden_states, self.weight, self.variance_epsilon)
    return output

def _mlp_forward_streamed(self, original_fwd, hidden_states, infer_state: InferState):
    # Use compute_stream only if spec_ret is active and pool initialized
    with torch.cuda.stream(infer_state.compute_stream):
        output = original_fwd(hidden_states) # Call original MLP forward
    return output

def _lm_head_forward_streamed(x, original_lm_head_forward, infer_state: InferState):
    with torch.cuda.stream(infer_state.compute_stream):
        logits = original_lm_head_forward(x[:, -1:, :])
    return logits

def _freekv_attn_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    infer_state: InferState = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    cur_id: int = self.layer_idx
    state = infer_state
    n_layers = state.n_layers

    with torch.cuda.stream(state.compute_stream):
        torch.cuda.nvtx.range_push(f"attention {cur_id}")
        if cur_id == 0:
            state.begin_forward(bsz, q_len)

        torch.cuda.nvtx.range_push(f"qkv_proj")
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        torch.cuda.nvtx.range_pop()

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        kvc = state.kv_caches[cur_id]
        budget = state.layer2budget[cur_id]

        torch.cuda.nvtx.range_push(f"RoPE")
        if self.config.rope_scaling is not None:
            assert self.config.rope_scaling["rope_type"] == "llama3"
            apply_llama31_rope_inplace(
                query_states.view(bsz*q_len, self.num_heads, self.head_dim),
                key_states.view(bsz*q_len, self.num_key_value_heads, self.head_dim),
                state.rope_indptr,
                state.rope_offsets,
                rope_scale=self.config.rope_scaling["factor"],
                rope_theta=self.config.rope_theta,
                low_freq_factor=self.config.rope_scaling["low_freq_factor"],
                high_freq_factor=self.config.rope_scaling["high_freq_factor"],
            )
        else:
            apply_rope_inplace(
                query_states.view(bsz*q_len, self.num_heads, self.head_dim),
                key_states.view(bsz*q_len, self.num_key_value_heads, self.head_dim),
                state.rope_indptr,
                state.rope_offsets,
                rope_scale=1.0,
                rope_theta=self.config.rope_theta,
            )
        torch.cuda.nvtx.range_pop()

        if q_len > 1:
            kvc.prefill_alloc_n_tokens(q_len, state.alloc_page)

        state.append_paged_kv_cache(cur_id, key_states, value_states)

        if q_len > 1:
            if budget is not None:
                with torch.cuda.stream(state.prefill_backup_stream):
                    state.prefill_backup_pages(cur_id)
                    evt = torch.cuda.Event()
                    evt.record(state.prefill_backup_stream)
                    state.prefill_backup_events[cur_id] = evt
                state.prefill_save_digests(cur_id, key_states)
                if state.echo_token:
                    state.echo_token_on_prefill(
                        cur_id,
                        query_states[:, -1:, ...].contiguous(),
                        key_states,
                        value_states,
                    )
            attn_output = state.prefill_sdpa(cur_id, query_states)
            infer_state.prefill_evict_extra_pages(
                cur_id, query_states[:, -1:, ...].contiguous()
            )
        else:
            if state.echo_token and budget is not None:
                state.echo_token_on_decode_append(cur_id, key_states, value_states)
            attn_page_ids = kvc.c2p
            if state.echo_token and budget is not None:
                # EchoKV native path: always decode via Echo runtime instead of
                # switching to legacy sparse path only after budget overflow.
                attn_output = state.decode_echo_token(cur_id, query_states)
            elif budget is not None and kvc.n_pages > budget:
                if state.echo_token:
                    attn_output = state.decode_echo_token(cur_id, query_states)
                elif state.spec_ret and cur_id not in NO_SPEC_RET_LAYER_SET:
                    pending_events = state.spec_ret_recall_status[cur_id]
                    if pending_events is not None:
                        evt1, evt2 = pending_events
                        state.compute_stream.wait_event(evt1)
                        state.compute_stream.wait_event(evt2)
                    else:
                        # first decoding step, need a recall before attn
                        state.estimate_select_recall(cur_id, query_states)
                    
                    to_corr = None
                    if state.corr is not None and state.last_step_q[cur_id] is not None:
                        torch.cuda.nvtx.range_push("cos")
                        if FORCE_CORR_RATE is not None:
                            to_corr_src = _force_to_corr[:bsz, :self.num_key_value_heads]
                            to_corr_gpu = state.to_corr_gpu[:bsz]
                            to_corr_gpu.copy_(to_corr_src, non_blocking=False)
                            state.need_corr_gpu.copy_(torch.any(to_corr_gpu))
                            need_corr = bool(state.need_corr_gpu.item())
                            if need_corr:
                                to_corr = state.to_corr_cpu_pinned[:bsz]
                                to_corr.copy_(to_corr_gpu, non_blocking=False)
                        else:
                            if cur_id in ALWAYS_CORR_LAYER_SET:
                                need_corr = True
                                to_corr = state.to_corr_cpu_pinned[:bsz]
                                to_corr.fill_(True)
                            elif state.corr_impl == "managed_cuda":
                                to_corr = state.to_corr_managed[:bsz]
                                need_corr = kernels.get_corr_managed_cuda(
                                    query_states,
                                    state.last_step_q[cur_id],
                                    self.num_key_value_heads,
                                    state.corr,
                                    to_corr,
                                )   # sync inside the kernel
                            else:
                                to_corr_res, need_corr_res = get_corr_torch_compile(
                                    query_states,
                                    state.last_step_q[cur_id],
                                    self.num_key_value_heads,
                                    state.corr,
                                )
                                need_corr = bool(need_corr_res.item())
                                if need_corr:
                                    to_corr = state.to_corr_cpu_pinned[:bsz]
                                    to_corr.copy_(to_corr_res, non_blocking=False)
                        torch.cuda.nvtx.range_pop()
                        state.corr_checks[cur_id] += 1
                        if need_corr:
                            state.corr_triggers[cur_id] += 1
                            eids, rids = state.estimate_select(cur_id, query_states)
                            state.recall(cur_id, eids, rids, blocking=True, need_recall_corr=to_corr)
                        else:
                            to_corr = None
                        
                    torch.cuda.nvtx.range_push("Fwd")
                    attn_output = state.decode_sdpa(cur_id, query_states, attn_page_ids)
                    torch.cuda.nvtx.range_pop()
                    recall_evt1, recall_evt2 = state.spec_ret_recall_events[cur_id]
                    state.spec_ret_recall_status[cur_id] = (recall_evt1, recall_evt2)
                    if state.corr is None or to_corr is None:
                        # not enable or not need correction
                        state.estimate_select_recall_pool(cur_id, query_states, 
                                                          recall_evt1, recall_evt2)
                    else:
                        # we have got eids and rids
                        state.recall(cur_id, eids, rids, blocking=False, 
                                     recall_evt1=recall_evt1, recall_evt2=recall_evt2,
                                     need_recall_corr=to_corr)

                    if state.corr is not None:
                        state.last_step_q[cur_id] = query_states
                else:
                    state.estimate_select_recall(cur_id, query_states)
                    attn_output = state.decode_sdpa(cur_id, query_states, attn_page_ids)
            else:
                # not reach the budget
                attn_output = state.decode_sdpa(cur_id, query_states, attn_page_ids)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        torch.cuda.nvtx.range_push("oproj")
        attn_output = self.o_proj(attn_output)
        torch.cuda.nvtx.range_pop()

        if not output_attentions:
            attn_weights = None

        if cur_id == n_layers - 1:
            state.end_forward(bsz, q_len)

        torch.cuda.nvtx.range_pop()
    
    return attn_output, attn_weights, past_key_value


def enable_offload(
    self: LlamaForCausalLM,
    dtype: torch.dtype,
    device: torch.device,
    page_size=32,
    infer_state: InferState = None,
    **kwargs,
):
    if infer_state is None:
        config = self.model.config
        infer_state = InferState(
            n_layers=config.num_hidden_layers,
            n_qo_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            page_size=page_size,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    if hasattr(self, "lm_head"):
        _lm_head_forward = self.lm_head.forward
        self.lm_head.forward = lambda x: \
            _lm_head_forward_streamed(x, _lm_head_forward, infer_state=infer_state)

    for mod in self.modules():
        mod_cls = str(mod.__class__)
        if "Attention" in mod_cls:
            mod.forward = (
                lambda mod: lambda *args, **kwargs: _freekv_attn_forward(
                    mod, *args, infer_state=infer_state, **kwargs
                )
            )(mod)
        elif "RMSNorm" in mod_cls:
            mod.forward = (
                lambda mod: lambda hidden_states: \
                _freekv_rms_norm_forward_streamed(mod, hidden_states, infer_state=infer_state)
            )(mod)
        elif "MLP" in mod_cls:
            _old_mlp_forward = mod.forward
            mod.forward = (
                lambda mod, old_fwd=_old_mlp_forward: lambda hidden_states: \
                _mlp_forward_streamed(mod, old_fwd, hidden_states, infer_state=infer_state)
            )(mod)

    _old_self_prepare_inputs_for_generation = self.prepare_inputs_for_generation
    _old_self_forward = self.forward

    @wraps(_old_self_prepare_inputs_for_generation)
    def _new_self_prepare_inputs_for_generation(input_ids, *args, **kwargs):
        kwargs["use_cache"] = False
        past_kv = kwargs.get("past_key_values", None)
        if past_kv is not None:
            assert past_kv == "dummy" or len(past_kv) == 0
            kwargs["past_key_values"] = None
            if past_kv == "dummy":  # not first prepare, decoding
                input_ids = input_ids[:, -1:]
        return _old_self_prepare_inputs_for_generation(input_ids, *args, **kwargs)

    @wraps(_old_self_forward)
    def _new_self_forward(*args, **kwargs):
        st = time.perf_counter()
        with torch.cuda.stream(infer_state.compute_stream):
            ret = _old_self_forward(*args, **kwargs)
        infer_state.compute_stream.synchronize()
        ret["past_key_values"] = "dummy"
        ed = time.perf_counter()
        self.tbt_stat_ms.append((ed-st)*1000)
        return ret

    self.tbt_stat_ms = []
    self.prepare_inputs_for_generation = _new_self_prepare_inputs_for_generation
    self.forward = _new_self_forward

    return infer_state
