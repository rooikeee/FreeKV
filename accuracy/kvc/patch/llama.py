from typing import Optional, Tuple
import os
import torch
from torch import nn

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    apply_rotary_pos_emb,
)
import types
from .utils import (
    reorder_linear_weights,
    reorder_full_attn_heads,
)

from .tuple_kv_cache import enable_tuple_kv_cache_for_llama

try:
    from flash_attn import flash_attn_func
    use_npu = False
except ImportError:
    flash_attn_func = None
    try:
        import torch_npu  # type: ignore  # noqa: F401
        use_npu = True
    except ImportError:
        use_npu = False
try:
    from .flashinfer_utils import apply_rope_inplace
except ImportError:
    apply_rope_inplace = None

from .dynamic_attention import (
    quest_arkv_attn, raas_attn
)

from .utils import flash_attn_maybe_npu, asym_quant_int8, asym_dequant_int8

def llama_dyn_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    indptr: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )
    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)

    # new data structure for past_key_value
    # past_key_value = (full_KV, dyn_KV, streaming_KV)
    # full_KV: (2 x bsz, num_full_key_value_heads, full_kv_seq_len, head_dim)
    # dyn_KV: (2 x bsz, num_dyn_key_value_heads, full_kv_seq_len, head_dim)
    # streaming_KV: (2 x bsz, num_streaming_key_value_heads, cache_size, head_dim)

    kv_seq_len = key_states.shape[1]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[1]

    is_raas = hasattr(self, "alpha")
    if is_raas:
        self.kv_seq_len = kv_seq_len

    if os.getenv("INPLACE_ROPE_OFF") is not None or apply_rope_inplace is None:
        # temp fix for multi-GPU ...
        cos, sin = position_embeddings
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
    else:
        apply_rope_inplace(self.config, query_states, key_states, 
                           position_ids[:, 0].contiguous(), indptr)

    if not hasattr(self, "full_attn_head_mask") or self.full_attn_head_mask is None:
        # Get the number of full/dyn/streaming kv heads
        self.full_attn_head_mask = self.full_attention_heads > 0.5
        self.num_full_attn_head = self.full_attn_head_mask.sum().item()
        if self.dyn_attention_heads is not None:
            self.dyn_attn_head_mask = self.dyn_attention_heads > 0.5
            self.num_dyn_attn_head = self.dyn_attn_head_mask.sum().item()
        else:
            self.num_dyn_attn_head = 0
        self.num_streaming_attn_head = (
            self.num_key_value_heads - self.num_full_attn_head - self.num_dyn_attn_head
        )

        # Get the number of full/dyn/streaming qo heads
        self.num_full_query_head = self.num_full_attn_head * self.num_key_value_groups
        self.num_dyn_query_head = self.num_dyn_attn_head * self.num_key_value_groups
        self.num_streaming_query_head = \
            self.num_heads - self.num_full_query_head - self.num_dyn_query_head

    full_key_states = key_states[:, :, : self.num_full_attn_head, :]
    full_value_states = value_states[:, :, : self.num_full_attn_head, :]

    dyn_key_states = key_states[
        :, :, self.num_full_attn_head: self.num_full_attn_head+self.num_dyn_attn_head, :
    ]
    dyn_value_states = value_states[
        :, :, self.num_full_attn_head: self.num_full_attn_head+self.num_dyn_attn_head, :
    ]

    streaming_key_states = key_states[
        :, :, self.num_full_attn_head+self.num_dyn_attn_head :, :
    ]
    streaming_value_states = value_states[
        :, :, self.num_full_attn_head+self.num_dyn_attn_head :, :
    ]
    # if self.layer_idx == 10:
    #     print(attention_mask, padding_mask)

    if past_key_value is not None:
        # reuse k, v, self_attention
        # [bsz*3, seq_len, num_*_heads, head_dim]
        past_full_KV = past_key_value[0]
        past_dyn_KV = past_key_value[1]
        past_streaming_KV = past_key_value[2]
        # [bsz, seq_len, num_full_attn_heads, head_dim]
        past_full_key_states = past_full_KV[:bsz]
        past_full_value_states = past_full_KV[bsz:]
        # [bsz, seq_len, num_dyn_attn_heads, head_dim]
        past_dyn_key_states = past_dyn_KV[:bsz]
        past_dyn_value_states = past_dyn_KV[bsz:]
        # [bsz, sink+recent, num_streaming_attn_heads, head_dim]
        past_streaming_key_states = past_streaming_KV[:bsz]
        past_streaming_value_states = past_streaming_KV[bsz:]
        if self.kv8:
            full_key_states = torch.cat([
                asym_dequant_int8(past_full_key_states, 
                                self.k_scale_full, self.k_zero_full, 
                                full_key_states.dtype), 
                full_key_states
            ], dim=1)
            full_value_states = torch.cat([
                asym_dequant_int8(past_full_value_states, 
                                self.v_scale_full, self.v_zero_full, 
                                full_value_states.dtype), 
                full_value_states
            ], dim=1)

            dyn_key_states = torch.cat([
                asym_dequant_int8(past_dyn_key_states,
                                self.k_scale_dyn, self.k_zero_dyn, 
                                target_dtype=dyn_key_states.dtype),
                dyn_key_states
            ], dim=1)
            dyn_value_states = torch.cat([
                asym_dequant_int8(past_dyn_value_states,
                                self.v_scale_dyn, self.v_zero_dyn, 
                                target_dtype=dyn_value_states.dtype),
                dyn_value_states
            ], dim=1)

            streaming_key_states = torch.cat([
                asym_dequant_int8(past_streaming_key_states, 
                                self.k_scale_streaming, self.k_zero_streaming, 
                                streaming_key_states.dtype), 
                streaming_key_states
            ], dim=1)
            streaming_value_states = torch.cat([
                asym_dequant_int8(past_streaming_value_states, 
                                self.v_scale_streaming, self.v_zero_streaming, 
                                streaming_value_states.dtype), 
                streaming_value_states
            ], dim=1)
        else:
            full_key_states = torch.cat([past_full_key_states, full_key_states], dim=1)
            full_value_states = torch.cat([past_full_value_states, full_value_states], dim=1)

            dyn_key_states = torch.cat([past_dyn_key_states, dyn_key_states], dim=1)
            dyn_value_states = torch.cat([past_dyn_value_states, dyn_value_states], dim=1)

            streaming_key_states = torch.cat([past_streaming_key_states, streaming_key_states], dim=1)
            streaming_value_states = torch.cat([past_streaming_value_states, streaming_value_states], dim=1)

    if q_len == kv_seq_len:
        # pre-filling: use flash attention
        if is_raas:
            attn_output = self.dynamic_attn(
                query_states,
                key_states,
                value_states,
                padding_mask,
            )
        else:
            attn_output = self._flash_attention_forward(
                query_states,
                key_states,
                value_states,
                padding_mask,
                q_len,
                dropout=0.0,
            )
    else:
        # decoding or continous filling
        placeholder = torch.tensor([], device=query_states.device, dtype=query_states.dtype)
        if self.num_full_attn_head > 0:
            full_query_states = query_states[:, :, : self.num_full_query_head, :]
            full_attn_output = self._flash_attention_forward(
                full_query_states,
                full_key_states,
                full_value_states,
                padding_mask,
                q_len,
                dropout=0.0,
            )
        else:
            full_attn_output = placeholder

        if self.num_dyn_attn_head > 0:
            dyn_query_states = query_states[
                :, :, self.num_full_query_head: self.num_full_query_head+self.num_dyn_query_head, :
            ]
            dyn_attn_output = self.dynamic_attn(
                dyn_query_states,
                dyn_key_states,
                dyn_value_states,
                padding_mask,
            )
        else:
            dyn_attn_output = placeholder

        if self.num_streaming_attn_head > 0:
            streaming_query_states = query_states[
                :, :, self.num_full_query_head+self.num_dyn_query_head :, :
            ]
            streaming_attn_output = self._flash_attention_forward(
                streaming_query_states,
                streaming_key_states,
                streaming_value_states,
                torch.cat([padding_mask[:, :self.sink_size], padding_mask[:, -self.recent_size:]],
                          dim=-1)[:, :streaming_key_states.shape[1]] if padding_mask is not None else None,
                q_len,
                dropout=0.0,
            )
        else:
            streaming_attn_output = placeholder

        # [bsz, 1, num_full_query_heads+num_streaming_query_heads, head_dim]
        attn_output = torch.cat(
            [full_attn_output, dyn_attn_output, streaming_attn_output], dim=2
        )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if is_raas:
        dyn_key_states = torch.cat([
            dyn_key_states[:, :self.sink_size + self.token_budget, ...],
            dyn_key_states[:, -self.recent_size:, ...],
        ], dim=1)
        dyn_value_states = torch.cat([
            dyn_value_states[:, :self.sink_size + self.token_budget, ...],
            dyn_value_states[:, -self.recent_size:, ...],
        ], dim=1)
    
    if self.kv8:
        if past_key_value is None:
            full_key_states, k_scale_full, k_zero_full = asym_quant_int8(
                full_key_states
            )
            full_value_states, v_scale_full, v_zero_full = asym_quant_int8(
                full_value_states
            )
            self.k_scale_full = k_scale_full
            self.v_scale_full = v_scale_full
            self.k_zero_full = k_zero_full
            self.v_zero_full = v_zero_full

            # fake decoding for update landmarks before quant
            _dyn_query_states = query_states[
                :, -1:, self.num_full_query_head: self.num_full_query_head+self.num_dyn_query_head, :
            ]
            if hasattr(self, "dynamic_attn"):
                _ = self.dynamic_attn(
                    _dyn_query_states,
                    dyn_key_states,
                    dyn_value_states,
                    padding_mask,
                    update_only=True
                )
            dyn_key_states, k_scale_dyn, k_zero_dyn = asym_quant_int8(
                dyn_key_states,
            )
            # dyn_key_states = dyn_key_states.view(bsz, q_len, self.num_dyn_attn_head, self.head_dim)
            dyn_value_states, v_scale_dyn, v_zero_dyn = asym_quant_int8(
                dyn_value_states,
            )
            # dyn_value_states = dyn_value_states.view(bsz, q_len, self.num_dyn_attn_head, self.head_dim)
            self.k_scale_dyn = k_scale_dyn
            self.v_scale_dyn = v_scale_dyn
            self.k_zero_dyn = k_zero_dyn
            self.v_zero_dyn = v_zero_dyn

            streaming_key_states, k_scale_streaming, k_zero_streaming = asym_quant_int8(
                streaming_key_states
            )
            streaming_value_states, v_scale_streaming, v_zero_streaming = asym_quant_int8(
                streaming_value_states
            )
            self.k_scale_streaming = k_scale_streaming
            self.v_scale_streaming = v_scale_streaming
            self.k_zero_streaming = k_zero_streaming
            self.v_zero_streaming = v_zero_streaming
        else:
            # always cat to make correct shape
            new_quant_full_key_states, k_scale_full, k_zero_full = asym_quant_int8(
                full_key_states[:, -q_len:, ...]
            )
            new_quant_full_value_states, v_scale_full, v_zero_full = asym_quant_int8(
                full_value_states[:, -q_len:, ...]
            )
            full_key_states = torch.cat([past_full_key_states, new_quant_full_key_states], dim=1)
            full_value_states = torch.cat([past_full_value_states, new_quant_full_value_states], dim=1)
            self.k_scale_full = torch.cat([self.k_scale_full, k_scale_full], dim=1)
            self.v_scale_full = torch.cat([self.v_scale_full, v_scale_full], dim=1)
            self.k_zero_full = torch.cat([self.k_zero_full, k_zero_full], dim=1)
            self.v_zero_full = torch.cat([self.v_zero_full, v_zero_full], dim=1)

            new_quant_dyn_key_states, k_scale_dyn, k_zero_dyn = asym_quant_int8(
                dyn_key_states[:, -q_len:, ...],
            )
            new_quant_dyn_value_states, v_scale_dyn, v_zero_dyn = asym_quant_int8(
                dyn_value_states[:, -q_len:, ...],
            )
            dyn_key_states = torch.cat([past_dyn_key_states, new_quant_dyn_key_states], dim=1)
            dyn_value_states = torch.cat([past_dyn_value_states, new_quant_dyn_value_states], dim=1)
            self.k_scale_dyn = torch.cat([self.k_scale_dyn, k_scale_dyn], dim=1)
            self.v_scale_dyn = torch.cat([self.v_scale_dyn, v_scale_dyn], dim=1)
            self.k_zero_dyn = torch.cat([self.k_zero_dyn, k_zero_dyn], dim=1)
            self.v_zero_dyn = torch.cat([self.v_zero_dyn, v_zero_dyn], dim=1)

            new_quant_streaming_key_states, k_scale_streaming, k_zero_streaming = asym_quant_int8(
                streaming_key_states[:, -q_len:, ...]
            )
            new_quant_streaming_value_states, v_scale_streaming, v_zero_streaming = asym_quant_int8(
                streaming_value_states[:, -q_len:, ...]
            )
            streaming_key_states = torch.cat([past_streaming_key_states, new_quant_streaming_key_states], dim=1)
            streaming_value_states = torch.cat([past_streaming_value_states, new_quant_streaming_value_states], dim=1)
            self.k_scale_streaming = torch.cat([self.k_scale_streaming, k_scale_streaming], dim=1)
            self.v_scale_streaming = torch.cat([self.v_scale_streaming, v_scale_streaming], dim=1)
            self.k_zero_streaming = torch.cat([self.k_zero_streaming, k_zero_streaming], dim=1)
            self.v_zero_streaming = torch.cat([self.v_zero_streaming, v_zero_streaming], dim=1)

    if streaming_key_states.shape[1] > self.recent_size + self.sink_size:
        recent_key_states = streaming_key_states[:, -self.recent_size :, :, :].clone()
        streaming_key_states[
            :, self.sink_size : self.sink_size + self.recent_size, :, :
        ].copy_(recent_key_states)
        streaming_key_states = streaming_key_states[
            :, : self.sink_size + self.recent_size, :, :
        ]

        recent_value_states = streaming_value_states[
            :, -self.recent_size :, :, :
        ].clone()
        streaming_value_states[
            :, self.sink_size : self.sink_size + self.recent_size, :, :
        ].copy_(recent_value_states)
        streaming_value_states = streaming_value_states[
            :, : self.sink_size + self.recent_size, :, :
        ]
        if self.kv8:
            recent_k_scale = self.k_scale_streaming[:, -self.recent_size :, ...].clone()
            self.k_scale_streaming[
                :, self.sink_size : self.sink_size + self.recent_size, :, :
            ].copy_(recent_k_scale)
            self.k_scale_streaming = self.k_scale_streaming[
                :, : self.sink_size + self.recent_size, :, :
            ]
            recent_k_zero = self.k_zero_streaming[:, -self.recent_size :, ...].clone()
            self.k_zero_streaming[
                :, self.sink_size : self.sink_size + self.recent_size, :, :
            ].copy_(recent_k_zero)
            self.k_zero_streaming = self.k_zero_streaming[
                :, : self.sink_size + self.recent_size, :, :
            ]

            recent_v_scale = self.v_scale_streaming[:, -self.recent_size :, ...].clone()
            self.v_scale_streaming[
                :, self.sink_size : self.sink_size + self.recent_size, :, :
            ].copy_(recent_v_scale)
            self.v_scale_streaming = self.v_scale_streaming[
                :, : self.sink_size + self.recent_size, :, :
            ]
            recent_v_zero = self.v_zero_streaming[:, -self.recent_size :, ...].clone()
            self.v_zero_streaming[
                :, self.sink_size : self.sink_size + self.recent_size, :, :
            ].copy_(recent_v_zero)
            self.v_zero_streaming = self.v_zero_streaming[
                :, : self.sink_size + self.recent_size, :, :
            ]
    
    # if self.layer_idx == 10:
    #     print(f"{streaming_key_states.shape=}, {self.k_scale_streaming.shape=}, {self.k_zero_streaming.shape=}")

    past_key_value = (
        (
            torch.cat([full_key_states, full_value_states], dim=0),
            torch.cat([dyn_key_states, dyn_value_states], dim=0),
            torch.cat([streaming_key_states, streaming_value_states], dim=0),
        )
        if use_cache
        else None
    )

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def enable_llama_dyn_attention(
    model: LlamaForCausalLM,
    full_heads: torch.Tensor,
    full_dyn_heads: torch.Tensor,
    sink_size: int,
    recent_size: int,
    method: str,
    config: dict,
    bsz=1,
):
    enable_tuple_kv_cache_for_llama(model)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    last_layer_attn = None
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        if last_layer_attn is not None:
            module.last_layer_attn = last_layer_attn
        layer_full_heads = torch.tensor(
            full_heads[idx], device=device, dtype=dtype
        )
        layer_full_dyn_heads = torch.tensor(
            full_dyn_heads[idx], device=device, dtype=dtype
        )
        layer_dyn_heads = layer_full_dyn_heads - layer_full_heads
        assert torch.all(layer_dyn_heads >= 0)
        if torch.all(layer_dyn_heads == 0):
            layer_dyn_heads = None

        module.forward = types.MethodType(
            llama_dyn_attention_forward, module
        )
        module.kv8 = config["kv8"]
        module.q_proj = reorder_linear_weights(
            module.q_proj,
            layer_full_heads,
            module.num_key_value_groups * module.head_dim,
            "out",
            layer_dyn_heads,
        )
        module.k_proj = reorder_linear_weights(
            module.k_proj,
            layer_full_heads,
            module.head_dim,
            "out",
            layer_dyn_heads,
        )
        module.v_proj = reorder_linear_weights(
            module.v_proj,
            layer_full_heads,
            module.head_dim,
            "out",
            layer_dyn_heads,
        )
        module.o_proj = reorder_linear_weights(
            module.o_proj,
            layer_full_heads,
            module.num_key_value_groups * module.head_dim,
            "in",
            layer_dyn_heads,
        )
        # NOTE: KV heads instead of QO heads
        layer_full_heads, layer_dyn_heads, num_full_attn_heads, num_dyn_attn_heads = \
            reorder_full_attn_heads(layer_full_heads, layer_dyn_heads)

        module.sink_size = sink_size
        module.recent_size = recent_size
        if method in ["quest", "arkv", "spec_ret"]:
            module.page_size = config["page_size"]
            module.token_budget = config["budget"] if idx >= config["skip_layer"] else -1
            module.GQA_policy = config["GQA_policy"]
            module.alloc_len = 512
            module.num_pages = 0    # === reset needed ===
            max_num_pages = 128 * 1024 // module.page_size
            if module.token_budget > 0:
                module.budget_ones = torch.ones((bsz, module.token_budget), 
                                                device=module.q_proj.weight.device)
            module.min_k = torch.empty((bsz, max_num_pages, num_dyn_attn_heads, module.head_dim), 
                                       device=module.q_proj.weight.device, dtype=dtype)
            module.max_k = torch.empty((bsz, max_num_pages, num_dyn_attn_heads, module.head_dim), 
                                       device=module.q_proj.weight.device, dtype=dtype)
            module.dynamic_attn = types.MethodType(quest_arkv_attn, module)
            if method == "arkv":
                module.page_rep = "arkv"
            else:   # for quest or spec_kv
                module.page_rep = config.get("page_rep", "quest")
            module.ll_token_budget = 0
            if method == "spec_ret":
                module.ll_token_budget = config["llb"]
                gqa_size = module.q_proj.weight.shape[0] // module.k_proj.weight.shape[0]
                module.spec_ret_steps = config["spec_ret_steps"]
                module.q_ptr = 0
                module.correct_sim = config["correct_sim"]
                module.corr_group = config["corr_group"]
                module.num_correct = 0  # === reset needed ===
                module.q_cache = torch.empty((bsz, module.spec_ret_steps, num_dyn_attn_heads*gqa_size, module.head_dim), 
                                            device=module.q_proj.weight.device, dtype=dtype)
        elif method == "raas":
            assert not module.kv8, "RaaS does not support int8 kv"
            module.kv_seq_len = 0
            module.dynamic_attn = types.MethodType(raas_attn, module)
            module.page_size = config["page_size"]
            module.token_budget = config["budget"] if idx >= config["skip_layer"] else -1
            module.alpha = config["raas_alpha"]
            module.page_budget = module.token_budget // module.page_size
            if module.token_budget > 0:
                module.budget_ones = torch.ones((bsz, module.token_budget), 
                                                device=module.q_proj.weight.device)
                # === reset needed =====
                module.num_pages = 0    
                module.page_timestamp = torch.zeros((bsz, module.page_budget, num_dyn_attn_heads), 
                                                    device=module.q_proj.weight.device,
                                                    dtype=torch.int32)
                # === const metadata ===
                module.page_indices = torch.arange(
                    module.page_size, device=module.q_proj.weight.device
                ).unsqueeze(0).unsqueeze(-1)  # [1, page_size, 1]
                module.head_indices = torch.arange(
                    num_dyn_attn_heads, device=module.page_indices.device
                ).view(1, 1, num_dyn_attn_heads)
                module.batch_ind = torch.arange(
                    bsz, device=module.page_indices.device
                ).view(bsz, 1, 1)
                # ======================
                module.cached_k = torch.empty((bsz, module.token_budget, num_dyn_attn_heads, module.head_dim), 
                                            device=module.q_proj.weight.device, dtype=dtype)
                module.cached_v = torch.empty((bsz, module.token_budget, num_dyn_attn_heads, module.head_dim), 
                                            device=module.q_proj.weight.device, dtype=dtype,)
                if module.page_size > 1:
                    module.min_k = torch.empty((bsz, module.page_budget, num_dyn_attn_heads, module.head_dim), 
                                            device=module.q_proj.weight.device, dtype=dtype)
                    module.max_k = torch.empty((bsz, module.page_budget, num_dyn_attn_heads, module.head_dim), 
                                            device=module.q_proj.weight.device, dtype=dtype)
        module.register_buffer(
            "full_attention_heads",
            layer_full_heads,
        )
        module.register_buffer(
            "dyn_attention_heads",
            layer_dyn_heads,
        )
        last_layer_attn = module

