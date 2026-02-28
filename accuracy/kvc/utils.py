import numpy as np
import os
import json


def load_attn_pattern(attn_load_dir):
    attn_load_path = attn_load_dir if not os.path.isdir(attn_load_dir) \
        else os.path.join(attn_load_dir, "full_attention_heads.tsv")
    full_attention_heads = np.loadtxt(
        attn_load_path,
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    
    sink_size = None
    recent_size = None
    if os.path.isdir(attn_load_dir):
        config = json.load(open(os.path.join(attn_load_dir, "config.json")))
        sink_size = config["sink_size"]
        recent_size = config["recent_size"]

    return full_attention_heads, sink_size, recent_size


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def sparsify_attention_heads(full_attention_heads, threshold=None, sparsity=None):
    # add a very small random noise to full_attention_heads to break ties
    full_attention_heads += np.random.uniform(0, 1e-6, full_attention_heads.shape)

    if sparsity is not None:
        # ignore the threshold and use the sparsity
        # set the sparsity small values to 0 and others to 1
        threshold = np.quantile(full_attention_heads, sparsity)
        if sparsity >= 1:
            # all heads are pruned
            threshold = 2
        if sparsity <= 0:
            # no heads are pruned
            threshold = -1
    else:
        assert threshold is not None, "Either threshold or sparsity must be provided"

    full_heads = (full_attention_heads >= threshold).astype(float)
    sparsity = 1 - np.mean(full_heads)

    return full_heads, sparsity
