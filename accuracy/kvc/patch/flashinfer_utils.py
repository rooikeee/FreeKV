from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm
import torch
import flashinfer
import types
from typing import Optional


def flashinfer_rmsnorm_forward(self, hidden_states):
    bsz, seq_len, hidden_size = hidden_states.size()
    weight = self.weight.to(hidden_states.device)
    hidden_states = flashinfer.norm.rmsnorm(
        hidden_states.view(bsz * seq_len, hidden_size),
        weight,
        eps=self.variance_epsilon,
    )
    return hidden_states.view(bsz, seq_len, hidden_size)


def enable_flashinfer_rmsnorm(model):
    print("Replacing RMSNorm with Flashinfer's RMSNorm")
    for name, module in model.named_modules():
        if isinstance(module, LlamaRMSNorm):
            module.forward = types.MethodType(flashinfer_rmsnorm_forward, module)
        elif isinstance(module, MistralRMSNorm):
            module.forward = types.MethodType(flashinfer_rmsnorm_forward, module)
    return model


def apply_rope_inplace(
    config,
    q: torch.Tensor,
    k: torch.Tensor,
    offsets: torch.Tensor,
    indptr: Optional[torch.Tensor] = None,
):
    bsz, q_len, num_heads, head_dim = q.size()
    num_kv_heads = k.shape[-2]
    if config.rope_scaling is not None:
        assert config.rope_scaling["rope_type"] == "llama3"
        flashinfer.apply_llama31_rope_inplace(
            q.view(bsz*q_len, num_heads, head_dim),
            k.view(bsz*q_len, num_kv_heads, head_dim),
            indptr,
            offsets,
            rope_scale=config.rope_scaling["factor"],
            rope_theta=config.rope_theta,
            low_freq_factor=config.rope_scaling["low_freq_factor"],
            high_freq_factor=config.rope_scaling["high_freq_factor"],
            interleave=False
        )
    else: 
        flashinfer.apply_rope_inplace(
            q.view(bsz*q_len, num_heads, head_dim),
            k.view(bsz*q_len, num_kv_heads, head_dim),
            indptr,
            offsets,
            rope_scale=1.0,
            rope_theta=config.rope_theta,
        )


def apply_rope_inplace0(
    q: torch.Tensor,
    k: torch.Tensor,
    offsets: torch.Tensor,
    rope_scale: float,
    rope_theta: float,
    indptr: Optional[torch.Tensor] = None,
):
    bsz, seq_len, num_heads, head_dim = q.size()
    _, _, num_kv_heads, _ = k.size()
    nnz = bsz * seq_len
    q = q.view(nnz, num_heads, head_dim)
    k = k.view(nnz, num_kv_heads, head_dim)
    if indptr is None:
        indptr = torch.tensor(
            [i * seq_len for i in range(bsz + 1)], dtype=torch.int32, device=q.device
        )
    if offsets.numel() == 1:
        offsets = offsets.expand(bsz).contiguous()
    flashinfer.rope.apply_rope_inplace(
        q,
        k,
        indptr,
        offsets,
        interleave=False,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    q = q.view(bsz, seq_len, num_heads, head_dim)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim)
    return q, k
