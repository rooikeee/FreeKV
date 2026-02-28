import os, argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import torch.nn.functional as F
from kvc.patch.tuple_kv_cache import enable_tuple_kv_cache
from kvc.utils import (
    load_attn_pattern,
    sparsify_attention_heads
)
from kvc.patch import (
    enable_dyn_attention, 
    QuestUpdater, RaaSUpdater, SpecRetUpdater
)
import torch.nn.functional as F
try:
    import torch_npu
    use_npu = True
except ImportError:
    use_npu = False
    pass

def parse_common_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_root_dir", type=str, required=True, help="Root directory for output files")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")

    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--method", type=str, default="full", 
                        choices=["full", "duo_attn", "razor", 
                                 "quest", "arkv", "spec_ret", "raas"],
                        help="KV cache eviction/selection method")
    parser.add_argument("--page_rep", type=str, default="quest", choices=["quest", "arkv"],
                        help="Page representation method")
    parser.add_argument("--GQA_policy", type=str, default="avgS",
                        choices=["maxQ", "avgQ", "maxS", "avgS", "maxSM", "avgSM"],
                        help="Grouped-query attention aggregation policy")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy decoding)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling threshold")
    parser.add_argument("--max_gen", type=int, default=8192,
                        help="Maximum number of tokens to generate")

    parser.add_argument("--data_from", type=int, default=None,
                        help="Starting index offset for data samples")
    parser.add_argument("--data_idx", type=int, default=None,
                        help="Specific data sample index to evaluate")
    parser.add_argument("--data_idx_to", type=int, default=None,
                        help="End index (exclusive) for data sample range")

    parser.add_argument(
        "--attn_load_dir", type=str, default="manual", help="attention pattern directory"
    )
    parser.add_argument("--sink", type=int, default=512,
                        help="Number of sink (initial) tokens to keep")
    parser.add_argument("--recent", type=int, default=512,
                        help="Number of recent tokens to keep")
    parser.add_argument("--budget", type=int, default=1024,
                        help="Token budget for dynamic KV cache selection")
    parser.add_argument("--page_size", type=int, default=32, 
                        help="For Quest, ArkVale, SpecRet and RaaS")

    parser.add_argument("--sparsity", type=float, default=1, 
                        help="Head-level sparsity, i.e., 1 - (ratio of full heads)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Attention score threshold for classifying full vs. streaming heads")

    parser.add_argument("--raas_alpha", type=float, default=1e-4,
                        help="Alpha parameter for RaaS method")

    parser.add_argument("--spec_ret_steps", type=int, default=1,
                        help="Number of past queries to use for SpecRet")
    parser.add_argument("--last_layer_budget", type=int, default=0,
                        help="Extra token budget for the last layer in SpecRet")
    parser.add_argument("--spec_ret_corr", type=float, default=None,
                        help="Cosine similarity threshold to trigger correction in SpecRet")
    parser.add_argument("--corr_group", type=str, default="avg", choices=["max", "avg"],
                        help="Aggregation method for correction grouping in SpecRet")

    parser.add_argument("--skip_layer", type=int, default=1,
                        help="Number of initial layers to skip (use full attention)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--repeat_bsz", type=int, default=None, 
                        help="set for efficiency eval")

    parser.add_argument("--kv8", action="store_true",
                        help="Use 8-bit quantized KV cache (FP16 instead of BF16)")
    return parser


def get_out_path(args, config, out_root_dir=None, mkdir=True):
    method = args.method
    dataset = args.dataset or ""
    model_name = args.model
    if out_root_dir is None:
        out_root_dir = args.out_root_dir
    if args.data_idx is not None or args.data_idx_to is not None:
        out_root_dir = str(Path(out_root_dir).parent / "res-test")
    out_dir = f"{out_root_dir}/{model_name}-{method}"
    if mkdir:
        os.makedirs(out_dir, exist_ok=True)
    if method in ["duo_attn", "razor"]:
        sparsity = config["sparsity"]
        sink = config["sink"]
        recent = config["recent"]
        out_path = f"{out_dir}/{dataset}-s{sink}-r{recent}-{sparsity:.2f}"
        attn_load = os.path.basename(args.attn_load_dir.rstrip("/"))
        pattern_id = attn_load if os.path.isdir(args.attn_load_dir) else attn_load.split(".")[0]
        if sparsity < 1:
            out_path += f"-{pattern_id}"
    elif method != "full":
        sink = config["sink"]
        recent = config["recent"]
        sparsity = config["sparsity"]
        attn_load = os.path.basename(args.attn_load_dir.rstrip("/"))
        pattern_id = attn_load if os.path.isdir(args.attn_load_dir) else attn_load.split(".")[0]
        out_path = (f"{out_dir}/{dataset}-s{sink}-r{recent}-{sparsity:.2f}-{pattern_id}")
        if method in ["quest", "raas", "arkv", "spec_ret"]:
            budget = config["budget"]
            page_size = config["page_size"]
            out_path += f"-p{page_size}-b{budget}"
            if method == "raas":
                alpha = config["raas_alpha"]
                out_path += f"-a{alpha}"
            else:
                GQA_policy = config["GQA_policy"]
                out_path += f"-{GQA_policy}"
                if method == "spec_ret":
                    spec_ret_steps = config["spec_ret_steps"]
                    llb = config["llb"]
                    correct_sim = config["correct_sim"]
                    corr_group = config["corr_group"]
                    out_path += f"-pQ{spec_ret_steps}-llb{llb}"
                    if correct_sim is not None:
                        out_path += f"-pQcs{correct_sim}"
                    if corr_group != "avg":
                        out_path += f"-cog{corr_group}"
        else:
            assert False and "Not covered method"
    else:
        out_path = f"{out_dir}/{dataset}"
    if args.skip_layer != 0:
        out_path += f"-skl{args.skip_layer}"
    if args.temperature != 0.0:
        out_path += f"-t{args.temperature}"
    if args.top_p != 1.0:
        out_path += f"-topp{args.top_p}"
    if args.max_gen != 8192:
        out_path += f"-Mg{args.max_gen}"
    if args.seed != 42:
        out_path += f"-seed{args.seed}"
    if args.data_from is not None:
        out_path += f"-from{args.data_from}"
    if args.data_idx_to is not None:
        out_path += f"-to{args.data_idx_to}"

    out_path += ".jsonl"
    print("Output to:", out_path)
    return out_path


def build_chat(tokenizer, prompt, model_name, to_token=True):
    if "ds-r1" in model_name or "skywork-or1" in model_name:
        return prompt + "<think>\n"
    elif "llama" in model_name:
        messages = [
            {"role": "user", "content": f"{prompt}"}
        ]
        if to_token:
            chat_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda" if not use_npu else "npu")
        else:
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=to_token, add_generation_prompt=True, return_tensors="pt"
            )
    elif "qwq" in model_name or "qwen" in model_name:
        messages = [
            {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba."},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "QwQ" in model_name:
        messages = [
            {"role": "user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return chat_prompt


def load_model_and_tokenizer(path, args):
    assert args.temperature >= 0 and args.top_p > 0.0 and args.top_p <= 1.0
    if args.method == "razor":
        assert args.sparsity is None and args.threshold == 0.5

    if "w8a8" in path:
        assert args.kv8, "W8A8 model must use kv8"

    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False if not "Skywork-OR1" in path else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        # trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not args.kv8 else torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        device_map = "auto",
    )

    generation_config = GenerationConfig.from_pretrained(path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    model = model.eval()

    config = {}
    method = args.method
    sink_size = args.sink
    recent_size = args.recent
    if method == "full":
        enable_tuple_kv_cache(model, kv8=args.kv8)
    else:
        if method in ["duo_attn", "razor"]:
            assert args.attn_load_dir is not None, "attn_load_dir must be provided"
            print(
                f"Loading attention pattern from {args.attn_load_dir} with sparsity {args.sparsity}"
            )
            full_attention_heads, _, _ = load_attn_pattern(
                args.attn_load_dir
            )
            assert sink_size is not None and recent_size is not None
            full_heads, true_sparsity = sparsify_attention_heads(
                full_attention_heads, args.threshold, args.sparsity
            )
            full_dyn_heads = full_heads
        elif args.sparsity == 1:
            print("All heads are dynamic heads")
            nl = model.config.num_hidden_layers
            nh = model.config.num_key_value_heads
            full_heads = np.zeros((nl, nh), dtype=float)
            full_dyn_heads = np.ones((nl, nh), dtype=float)
            true_sparsity = 1.0
        else:
            assert args.attn_load_dir is not None, "attn_load_dir must be provided"
            print(
                f"Loading attention pattern from {args.attn_load_dir} with sparsity {args.sparsity}"
            )
            full_attention_heads, _, _ = load_attn_pattern(
                args.attn_load_dir
            )
            assert sink_size is not None and recent_size is not None
            full_heads, true_sparsity = sparsify_attention_heads(
                full_attention_heads, args.threshold, args.sparsity
            )
            full_dyn_heads = np.ones_like(full_heads)

        print(f"True sparsity: {true_sparsity}")

        config["kv8"] = args.kv8
        config["sparsity"] = true_sparsity
        config["sink"] = sink_size
        config["recent"] = recent_size
        config["skip_layer"] = args.skip_layer
        config["page_rep"] = args.page_rep
        print(f"Enabling {method} evaluation using sink size {sink_size} recent size {recent_size}, "
              f"skip {args.skip_layer} layers")
        if method in ["quest", "arkv", "spec_ret"]:
            print(f"Budget: {args.budget}, page size: {args.page_size}, GQA policy: {args.GQA_policy}")
            assert recent_size % args.page_size == 0
            assert args.budget % args.page_size == 0
            config["budget"] = args.budget
            config["page_size"] = args.page_size
            config["GQA_policy"] = args.GQA_policy
            if method == "spec_ret":
                config["spec_ret_steps"] = args.spec_ret_steps
                config["llb"] = args.last_layer_budget
                config["correct_sim"] = args.spec_ret_corr
                config["corr_group"] = args.corr_group
                print(f"spec_ret_steps {args.spec_ret_steps}, last layer {args.last_layer_budget} tokens")
                if args.spec_ret_corr is not None:
                    print(f"Correct when cos(q, past_q) < {args.spec_ret_corr}")
        elif method == "raas":
            print(f"Budget: {args.budget}, page size: {args.page_size}, alpha {args.raas_alpha}")
            assert recent_size % args.page_size == 0
            assert sink_size % args.page_size == 0
            assert args.budget % args.page_size == 0
            config["budget"] = args.budget
            config["page_size"] = args.page_size
            config["raas_alpha"] = args.raas_alpha

        enable_dyn_attention(
            model,
            full_heads,
            full_dyn_heads,
            sink_size,
            recent_size,
            # for dynamic heads
            method,
            config,
            args.batch_size if args.repeat_bsz is None else args.repeat_bsz,
        )
    
    if method in ["quest", "arkv"]:
        step_updater = QuestUpdater(model)
    elif method == "spec_ret":
        step_updater = SpecRetUpdater(model)
    elif method == "raas":
        step_updater = RaaSUpdater(model)
    else:
        step_updater = None

    return model, tokenizer, step_updater, eos_token_ids, config

def sample_token(output, temperature, top_p):
    logits = output.logits[:, -1, :]
    if  temperature <= 0:
        pred_token_idx = logits.argmax(dim=-1).unsqueeze(1)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            probs[indices_to_remove] = 0.0
            renormalized_probs = probs / (torch.sum(probs, dim=-1, keepdim=True) + 1e-9)
            final_probs = renormalized_probs
        else:
            final_probs = probs

        pred_token_idx = torch.multinomial(final_probs, num_samples=1)
    return pred_token_idx