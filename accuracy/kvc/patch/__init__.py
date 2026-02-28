from .llama import enable_llama_dyn_attention
# from .w8a8.llama_w8a8 import LlamaForCausalLM as LlamaForCausalLM_W8A8
# from .w8a8.config import ModelConfig as W8A8ModelConfig

import numpy as np
import os
import torch

from .step_update import *


def enable_dyn_attention(
    model,
    full_heads,
    full_dyn_heads,
    sink_size,
    recent_size,
    method,
    config,
    bsz=1,
):
    if "llama" in model.config.model_type or "qwen2" in model.config.model_type:
        enable_llama_dyn_attention(
            model,
            full_heads,
            full_dyn_heads,
            sink_size,
            recent_size,
            method,
            config,
            bsz,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def load_full_attention_heads(load_dir, filename="full_attention_heads.tsv"):
    full_attention_heads = np.loadtxt(
        os.path.join(load_dir, filename),
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    full_attention_heads = torch.tensor(full_attention_heads, dtype=torch.float32)
    return full_attention_heads
