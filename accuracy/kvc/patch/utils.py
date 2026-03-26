import os
import torch
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


@torch.no_grad()
def reorder_linear_weights(
    linear_module: torch.nn.Linear,
    full_attention_heads: torch.Tensor,
    repeat_num,
    reorder_channel,
    dyn_attention_heads: torch.Tensor = None,
):
    assert reorder_channel in ["in", "out"]
    full_attention_heads = torch.repeat_interleave(
        full_attention_heads, repeats=repeat_num
    ).to(linear_module.weight.device)
    full_attn_mask = full_attention_heads > 0.5
    if dyn_attention_heads is not None:
        dyn_attention_heads = torch.repeat_interleave(
            dyn_attention_heads, repeats=repeat_num
        ).to(linear_module.weight.device)
        dyn_attn_mask = dyn_attention_heads > 0.5

    if reorder_channel == "in":
        weight1 = linear_module.weight.data[:, full_attn_mask]
        if dyn_attention_heads is None:
            weight2 = linear_module.weight.data[:, ~full_attn_mask]
            reordered_weight = torch.cat([weight1, weight2], dim=1)
        else:
            weight2 = linear_module.weight.data[:, dyn_attn_mask]
            weight3 = linear_module.weight.data[:, ~(full_attn_mask+dyn_attn_mask)]
            reordered_weight = torch.cat([weight1, weight2, weight3], dim=1)
    else:
        weight1 = linear_module.weight.data[full_attn_mask, :]
        if dyn_attention_heads is None:
            weight2 = linear_module.weight.data[~full_attn_mask, :]
            reordered_weight = torch.cat([weight1, weight2], dim=0)
        else:
            weight2 = linear_module.weight.data[dyn_attn_mask, :]
            weight3 = linear_module.weight.data[~(full_attn_mask+dyn_attn_mask), :]
            reordered_weight = torch.cat([weight1, weight2, weight3], dim=0)
    linear_module.weight.data = reordered_weight
    # for linear modules with bias
    if linear_module.bias is not None:
        bias1 = linear_module.bias.data[full_attn_mask]
        if dyn_attention_heads is None:
            bias2 = linear_module.bias.data[~full_attn_mask]
            reordered_bias = torch.cat([bias1, bias2], dim=0)
        else:
            bias2 = linear_module.bias.data[dyn_attn_mask]
            bias3 = linear_module.bias.data[~(full_attn_mask+dyn_attn_mask)]
            reordered_bias = torch.cat([bias1, bias2, bias3], dim=0)
        linear_module.bias.data = reordered_bias

    return linear_module


@torch.no_grad()
def reorder_full_attn_heads(
    full_attention_heads: torch.Tensor,
    dyn_attention_heads: torch.Tensor = None,
):
    full_attn_mask = full_attention_heads > 0.5
    num_full_attn_heads = full_attn_mask.sum().item()
    full_attention_heads[:num_full_attn_heads] = 1
    full_attention_heads[num_full_attn_heads:] = 0

    num_dyn_attn_heads = 0
    if dyn_attention_heads is not None:
        dyn_attn_mask = dyn_attention_heads > 0.5
        num_dyn_attn_heads = dyn_attn_mask.sum().item()
        dyn_attention_heads = torch.zeros_like(dyn_attention_heads)
        dyn_attention_heads[num_full_attn_heads: num_full_attn_heads+num_dyn_attn_heads] = 1

    return full_attention_heads, dyn_attention_heads, num_full_attn_heads, num_dyn_attn_heads


def flash_attn_maybe_npu(
    query_states,
    key_states,
    value_states,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True,
):
    disable_flash_attn = os.getenv("DISABLE_FLASH_ATTN", "0") == "1"
    if (not disable_flash_attn) and flash_attn_func is not None:
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=True,
        )
    else:
        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)

        # SDPA requires matching head counts; for GQA/MQA we expand KV heads.
        qh = q.shape[1]
        kh = k.shape[1]
        if qh != kh:
            if qh % kh != 0:
                raise RuntimeError(
                    f"SDPA head mismatch: q_heads={qh}, kv_heads={kh} (not divisible)."
                )
            n_rep = qh // kh
            k = torch.repeat_interleave(k, repeats=n_rep, dim=1)
            v = torch.repeat_interleave(v, repeats=n_rep, dim=1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p, 
            scale=softmax_scale,
            is_causal=causal
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output

# def per_token_int8(x: torch.Tensor):
#     x_min = x.amin(dim=-1, keepdim=True)
#     x_max = x.amax(dim=-1, keepdim=True)
#     scale = torch.maximum(x_max.abs(), x_min.abs()) / 127.0
#     x_int8 = torch.clamp((x / scale).round(), -127, 127).to(torch.int8)
#     return x_int8, scale

# def per_token_int8_dequant(x_int8: torch.Tensor,    # [B, T, H, D]
#                            scale: torch.Tensor,     # [B, T, H, 1]
#                            target_dtype=torch.bfloat16):
#     return (x_int8.float() * scale).to(target_dtype)

def asym_quant_int8(x: torch.Tensor, dim=-1):
    if dim is None:
        # per-tensor
        x_min = x.min()
        x_max = x.max()
    else:
        # per-token / per-channel
        x_min = x.amin(dim=dim, keepdim=True)
        x_max = x.amax(dim=dim, keepdim=True)

    qmin, qmax = -128.0, 127.0
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = qmin - (x_min / scale)
    zero_point = torch.round(zero_point).clamp(qmin, qmax)

    x_int8 = torch.round(x / scale + zero_point).clamp(qmin, qmax).to(torch.int8)
    return x_int8, scale, zero_point

def asym_dequant_int8(x_int8: torch.Tensor, 
                      scale: torch.Tensor, 
                      zero_point: torch.Tensor,
                      target_dtype=torch.bfloat16):
    d = (x_int8.to(torch.float32) - zero_point) * scale
    return d.to(target_dtype)
