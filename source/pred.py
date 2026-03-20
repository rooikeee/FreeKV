from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from freekv import adapter
import torch
from tqdm import tqdm
import json, os, argparse, random
from datasets import load_dataset
import numpy as np
import time

c = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{c[0]}.{c[1]}"

BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
SEP    = "─" * 100

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def build_chat(tokenizer, prompt, model_name):
    model_name_lower = model_name.lower()
    if "ds-r1" in model_name_lower:
        return prompt + "<think>\n"
    if "llama" in model_name_lower:
        messages = [
            {"role": "user", "content": f"{prompt}"}
        ]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
    if "qwen" in model_name_lower:
        messages = [
            {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if "qwq" in model_name_lower:
        messages = [
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def generate_once(model, input_ids, max_gen, temperature, eos_token_ids, pad_token_id):
    if temperature > 0:
        return model.generate(
            input_ids,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=True,
            temperature=temperature,
            eos_token_id=eos_token_ids,
            pad_token_id=pad_token_id,
            past_key_values=None,
        )
    return model.generate(
        input_ids,
        max_new_tokens=max_gen,
        do_sample=False,
        eos_token_id=eos_token_ids,
        pad_token_id=pad_token_id,
        past_key_values=None,
    )


def simplify_text_preview(text, max_tokens=100):
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", type=str, default=None, help="Model name (key in config/model2path.json)")
    parser.add_argument("--dataset", type=str, required=True, choices=["AIME24", "gov_report", "lgbench"], help="Evaluation dataset to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--max_length", type=int, default=32000, help="Max input token length (longer inputs are truncated from the middle)")
    parser.add_argument(
        "--expand_prompt_to_max_length",
        action="store_true",
        help="For prompts in (8192, max_length), repeat then trim to max_length (stress mode).",
    )
    parser.add_argument("--max_gen", type=int, default=8192, help="Max number of new tokens to generate")
    parser.add_argument("--data_idx", type=int, default=None, help="Single data index to evaluate")
    parser.add_argument("--data_idx_to", type=int, default=None, help="Evaluate data from index 0 to this index (exclusive)")

    parser.add_argument("--sink", type=int, default=512, help="Number of sink tokens to keep")
    parser.add_argument("--recent", type=int, default=512, help="Number of recent tokens to keep")
    parser.add_argument("--budget", type=int, default=2048, help="Total token budget including sink and recent")
    parser.add_argument("--page_size", type=int, default=32, help="KV cache retrieval page size")
    parser.add_argument("--cpu_layout", type=str, default="HND", choices=["NHD", "HND"], help="CPU KV cache memory layout")
    parser.add_argument("--spec_ret", action="store_true", help="Enable speculative retrieval")
    parser.add_argument("--repeat_bsz", type=int, default=1, help="Repeat input to simulate batch size")
    parser.add_argument("--thread_pool", type=int, default=2, help="Number of threads in the recall thread pool")
    parser.add_argument("--n_recall_stream", type=int, default=2, help="Number of CUDA streams for recall")
    parser.add_argument("--recall_impl", type=str, default="cuda_cpy",
                        choices=["arkvale", "torch_cpy", "cuda_cpy"], help="Recall implementation")
    parser.add_argument("--corr", type=float, default=None, help="Correction threshold (cosine similarity); None to disable")
    parser.add_argument("--sel_policy", type=str, default="topk",
                        choices=["topk", "echokv_token"],
                        help="Selection policy")
    parser.add_argument("--echo_num_anchors", type=int, default=64,
                        help="Number of seed anchors for EchoKV-token")
    parser.add_argument("--echo_anchor_head_sample", type=int, default=0,
                        help="Sampled KV heads for anchor scoring (0=all heads)")
    parser.add_argument("--echo_attn_backend", type=str, default="sdpa",
                        choices=["sdpa", "flashinfer", "flash_attn"],
                        help="Attention backend for EchoKV-token decode")
    parser.add_argument("--echo_flash_mode", type=str, default="fused_fast",
                        choices=["fused_fast", "split_overlap"],
                        help="Flash-attn decode mode: fused_fast (lower latency) or split_overlap")
    parser.add_argument("--echo_shared_batch", dest="echo_shared_batch", action="store_true",
                        help="Use batch-0 shared anchors/recall for repeated batch decoding")
    parser.add_argument("--echo_no_shared_batch", dest="echo_shared_batch", action="store_false",
                        help="Disable shared-batch optimization for EchoKV-token")
    parser.set_defaults(echo_shared_batch=True)
    parser.add_argument("--echo_disable_cuda_token_recall", action="store_true",
                        help="Force disable custom cuda token-recall kernel")
    parser.add_argument("--echo_disable_triton_qk_select", action="store_true",
                        help="Disable Triton QK page-argmax selector for EchoKV-token")
    parser.add_argument("--echo_disable_triton_flash_attn", action="store_true",
                        help="Disable Triton fused decode-attention(+anchor) for EchoKV-token")
    parser.add_argument("--echo_allow_flash_fallback", action="store_true",
                        help="Allow fallback when Triton flash path is unavailable")
    parser.add_argument("--disable_profile_timing", action="store_true",
                        help="Disable CUDA event timing for select/recall/attn")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup generation rounds before timing")

    return parser.parse_args(args)


def load_model_and_tokenizer(path):
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )
    generation_config = GenerationConfig.from_pretrained(path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]
    model = (
        AutoModelForCausalLM
        .from_pretrained(path, torch_dtype=dtype, device_map=dev)
        .eval()
    )
    page_size = args.page_size
    token_budgets = args.budget
    page_budgets = token_budgets // page_size
    n_sink_pages = args.sink // page_size
    n_win_pages = args.recent // page_size
    print(f"\n{CYAN}{SEP}")
    print(f"  KV Cache Config: token_budget={token_budgets}, page_budget={page_budgets}, "
          f"page_size={page_size}, sink={args.sink}, recent={args.recent}")
    print(f"  Echo Config: sel_policy={args.sel_policy}, seed_anchors={args.echo_num_anchors}, "
          f"shared_batch={args.echo_shared_batch}, anchor_head_sample={args.echo_anchor_head_sample}, "
          f"attn_backend={args.echo_attn_backend}, "
          f"flash_mode={args.echo_flash_mode}, "
          f"triton_qk_select={(not args.echo_disable_triton_qk_select)}, "
          f"triton_flash_attn={(not args.echo_disable_triton_flash_attn)}, "
          f"triton_flash_strict={(not args.echo_allow_flash_fallback)}")
    print(f"{SEP}{RESET}\n")
    if token_budgets > 0:
        infer_state = adapter.enable_offload(
            model, 
            dtype=dtype, 
            device=dev, 
            page_size=page_size,
            page_budgets=page_budgets,
            page_topks=page_budgets-1,
            n_sink_pages=n_sink_pages,
            n_win_pages=n_win_pages,
            n_max_bytes=6 * (1 << 30),
            n_max_cpu_bytes=20 * (1 << 30),
            group_size=1,
            cpu_layout=args.cpu_layout,
            spec_ret=args.spec_ret,
            thread_pool_size=args.thread_pool,
            n_recall_stream=args.n_recall_stream,
            recall_impl=args.recall_impl,
            corr=args.corr,
            sel_policy=args.sel_policy,
            echo_num_anchors=args.echo_num_anchors,
            echo_anchor_head_sample=args.echo_anchor_head_sample,
            echo_attn_backend=args.echo_attn_backend,
            echo_flash_mode=args.echo_flash_mode,
            echo_shared_batch=args.echo_shared_batch,
            echo_use_cuda_token_recall=(not args.echo_disable_cuda_token_recall),
            echo_use_triton_qk_select=(not args.echo_disable_triton_qk_select),
            echo_use_triton_flash_attn=(not args.echo_disable_triton_flash_attn),
            echo_require_triton_flash=(not args.echo_allow_flash_fallback),
            profile_timing=(not args.disable_profile_timing),
        )
    else:
        assert not args.spec_ret
        infer_state = adapter.enable_offload(
            model, 
            dtype=dtype, 
            device=dev, 
            page_size=page_size,
            page_budgets=None, # page_budgets=None means "full" (no eviction & recall)
            n_max_bytes=6 * (1 << 30),
            n_max_cpu_bytes=40 * (1 << 30),
            group_size=1,
        )

    return model, tokenizer, eos_token_ids, infer_state


def get_pred(
    model,
    tokenizer,
    eos_token_ids,
    infer_state,
    data,
    answer_field_id,
    max_length,
    max_gen,
    prompt_format,
    model_name,
    temperature,
    warmup,
    expand_prompt_to_max_length,
):
    preds = []
    for json_obj in tqdm(data):
        prep_t0 = time.perf_counter()
        if prompt_format is not None:
            prompt = prompt_format.format(**json_obj)
        else:
            prompt = json_obj["prompt"]

        chat_prompt = build_chat(tokenizer, prompt, model_name)
        if isinstance(chat_prompt, str):
            input_ids = tokenizer(chat_prompt, truncation=False, return_tensors="pt").input_ids
        else:
            input_ids = chat_prompt

        src_len = int(input_ids.shape[-1])
        if expand_prompt_to_max_length and src_len > 8192 and src_len < max_length:
            print(
                f"{YELLOW}[PromptExpand] input_len={src_len} in (8192, {max_length}), "
                f"expanding to ~{max_length} for stress mode.{RESET}"
            )
            rep = max_length // src_len + 1
            input_ids = input_ids.repeat(1, rep)
            src_len = int(input_ids.shape[-1])

        if src_len > max_length:
            left = max_length // 2
            right = max_length - left
            input_ids = torch.cat([input_ids[:, :left], input_ids[:, -right:]], dim=-1)

        input = input_ids.to("cuda")
        prompt_length = input.shape[-1]
        prep_ms = (time.perf_counter() - prep_t0) * 1000.0
        print(
            f"{CYAN}[Input] prompt_length={prompt_length}, dtype={input.dtype}, "
            f"device={input.device}, prep={prep_ms:.1f}ms{RESET}"
        )
        input = input.repeat(args.repeat_bsz, 1)
        with torch.no_grad():
            if warmup > 0:
                print(f"{YELLOW}[Warmup] Running {warmup} warmup round(s)...{RESET}")
                for _ in range(warmup):
                    _ = generate_once(
                        model, input, max_gen, temperature, eos_token_ids, tokenizer.eos_token_id
                    )
                print(f"{YELLOW}[Warmup] Done.{RESET}")
                model.tbt_stat_ms.clear()
            infer_state.reset_perf_stats()

            st = time.perf_counter()
            output = generate_once(
                model, input, max_gen, temperature, eos_token_ids, tokenizer.eos_token_id
            )
            ed = time.perf_counter()
        perf = infer_state.get_perf_stats(synchronize=True, reset=False)
        
        gen_token = [len(o) - len(i) for o, i in zip(output, input)]
        prefill_ms = model.tbt_stat_ms[0] if model.tbt_stat_ms else 0.0
        decode_ms = model.tbt_stat_ms[1:]
        avg_tbt = sum(decode_ms) / len(decode_ms) if decode_ms else 0.0
        decode_total_ms = sum(decode_ms) if decode_ms else 0.0
        select_ms = perf["select_ms"]
        recall_ms = perf["recall_ms"]
        attn_ms = perf["attn_ms"]
        pack_ms = perf.get("pack_ms", 0.0)
        other_decode_ms = max(0.0, decode_total_ms - select_ms - recall_ms - attn_ms - pack_ms)

        print(f"\n{GREEN}{SEP}")
        print(f"  [{model_name}] Generation Summary")
        print(f"{SEP}{RESET}")
        print(f"  Tokens generated : {gen_token}")
        print(f"  Total time       : {sum(model.tbt_stat_ms)/1000:.2f}s")
        print(f"  Prefill time     : {prefill_ms/1000:.2f}s")
        print(f"  Decode time      : {sum(decode_ms)/1000:.2f}s")
        print(f"  Avg TBT (decode) : {avg_tbt:.2f} ms")
        print(f"  Select time      : {select_ms/1000:.2f}s ({perf['select_calls']} calls)")
        print(f"  Recall time      : {recall_ms/1000:.2f}s ({perf['recall_calls']} calls)")
        print(f"  Assemble time    : {pack_ms/1000:.2f}s ({perf.get('pack_calls', 0)} calls)")
        print(f"  Decode attn time : {attn_ms/1000:.2f}s ({perf['attn_calls']} calls)")
        print(f"  Decode other     : {other_decode_ms/1000:.2f}s")
        print(f"{GREEN}{SEP}{RESET}\n")
        for b in range(len(output)):
            pred = tokenizer.decode(output[b], skip_special_tokens=True)
            pred_only_output = tokenizer.decode(output[b][len(input[b]):], skip_special_tokens=True)
            print(f"{BOLD}[Batch {b}] Output preview:{RESET} {simplify_text_preview(pred_only_output, max_tokens=100)}")
            preds.append(
                {
                    "input:": prompt,
                    "pred": pred,
                    "answer": json_obj[answer_field_id] if answer_field_id is not None else "",
                    "input_len": len(input[b]),
                    "output_len": len(output[b]),
                }
            )
    return preds


if __name__ == "__main__":
    args = parse_args()
    if args.sel_policy == "echokv_token":
        # EchoKV-token path uses token-wise runtime recall, not FreeKV page recall.
        if args.spec_ret:
            print(
                f"{YELLOW}[EchoKV] --spec_ret is ignored for sel_policy=echokv_token; disabling it.{RESET}"
            )
            args.spec_ret = False
        if args.corr is not None:
            print(
                f"{YELLOW}[EchoKV] --corr is ignored for sel_policy=echokv_token; clearing it.{RESET}"
            )
            args.corr = None
        print(
            f"{YELLOW}[EchoKV] FreeKV recall args (--recall_impl/--cpu_layout) are kept for "
            f"legacy cache wiring, but decode recall uses Echo token-wise recall kernels.{RESET}"
        )
    seed_everything(args.seed)
    model2path = json.load(open("config/model2path.json", "r"))

    model_name = args.model
    max_length = args.max_length
    max_gen = args.max_gen
    model, tokenizer, eos_token_ids, infer_state = load_model_and_tokenizer(model2path[model_name])

    dataset = args.dataset
    ds_dir = "eval/datasets"
    answer_field_id = None
    prompt_format = None
    if dataset == "gov_report":
        # data_idx: 59, length:32726
        ds_path = f"{ds_dir}/gov_report.jsonl"
        prompt_format = "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    elif dataset == "lgbench":
        ds_path = f"{ds_dir}/longgenbench.json"
    elif dataset == "AIME24":
        answer_field_id = "answer"
        ds_path = f"{ds_dir}/aime_2024.jsonl"
        dataset2prompt = json.load(open("eval/o1/config/dataset2prompt.json", "r"))
        prompt_format = dataset2prompt[dataset]

    data = load_dataset("json", data_files=ds_path, split="train")

    if args.data_idx is not None:
        data = data.select(range(args.data_idx, args.data_idx+1))
    elif args.data_idx_to is not None:
        data = data.select(range(0, args.data_idx_to))
    preds = get_pred(
        model,
        tokenizer,
        eos_token_ids,
        infer_state,
        data,
        answer_field_id,
        max_length,
        max_gen,
        prompt_format,
        model_name,
        args.temperature,
        args.warmup,
        args.expand_prompt_to_max_length,
    )

    out_dir = f"tmp_res/{model_name}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = f"{out_dir}/{dataset}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")


# Original ArkVale
# python eval/o1/pred.py --model ds-r1-qwen-7b --dataset gov_report --temperature 0.0 --max_gen 1024 --data_idx 0 --warmup 0 --recall_impl arkvale

# Without correction
# python eval/o1/pred.py --model ds-r1-qwen-7b --dataset gov_report --temperature 0.0 --max_gen 1024 --data_idx 0 --warmup 0 --recall_impl cuda_cpy --cpu_layout HND --spec_ret

# With correction
# python eval/o1/pred.py --model ds-r1-qwen-7b --dataset gov_report --temperature 0.0 --max_gen 1024 --data_idx 0 --warmup 0 --recall_impl cuda_cpy --cpu_layout HND --spec_ret --corr 0.9
