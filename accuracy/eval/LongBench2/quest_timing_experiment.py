import argparse
import json
import os
from typing import List, Tuple

import torch
from datasets import load_dataset

from eval.util import parse_common_args, build_chat, load_model_and_tokenizer

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "config"
)
model_map = json.loads(
    open(os.path.join(_CONFIG_DIR, "model2path.json"), encoding="utf-8").read()
)
template_0shot = open("eval/LongBench2/prompts/0shot.txt", encoding="utf-8").read()


def parse_lengths(lengths: str) -> List[int]:
    out = []
    for token in lengths.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def build_prompt(item: dict, context: str) -> str:
    return (
        template_0shot.replace("$DOC$", context.strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )


def build_input_ids(prompt: str, model_name: str, tokenizer) -> torch.Tensor:
    chat_prompt = build_chat(tokenizer, prompt, model_name)
    if isinstance(chat_prompt, str):
        return tokenizer(chat_prompt, truncation=False, return_tensors="pt").to(
            "cuda"
        ).input_ids
    return chat_prompt


def get_input_len(prompt: str, model_name: str, tokenizer) -> int:
    return int(build_input_ids(prompt, model_name, tokenizer).shape[-1])


def truncate_context_to_target_len(
    item: dict,
    target_len: int,
    model_name: str,
    tokenizer,
) -> Tuple[str, int]:
    context_ids = tokenizer.encode(item["context"].strip(), add_special_tokens=False)
    lo, hi = 0, len(context_ids)
    best_context = ""
    best_len = -1

    while lo <= hi:
        mid = (lo + hi) // 2
        candidate_context = tokenizer.decode(
            context_ids[:mid], skip_special_tokens=True
        )
        candidate_prompt = build_prompt(item, candidate_context)
        candidate_len = get_input_len(candidate_prompt, model_name, tokenizer)
        if candidate_len <= target_len:
            best_context = candidate_context
            best_len = candidate_len
            lo = mid + 1
        else:
            hi = mid - 1

    if best_len < 0:
        raise RuntimeError(
            f"Cannot fit target_len={target_len}; even empty context exceeds it."
        )
    return best_context, best_len


def run_one_case(
    prompt: str,
    model_name: str,
    model,
    tokenizer,
    step_updater,
    decode_tokens: int,
) -> Tuple[dict, int]:
    input_ids = build_input_ids(prompt, model_name, tokenizer)
    decode_steps = 0
    with torch.no_grad():
        if step_updater is not None:
            step_updater.reset(input_ids)
        output = model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        for _ in range(decode_tokens):
            output = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            if step_updater is not None:
                step_updater.update(pred_token_idx.item())
            decode_steps += 1
    if step_updater is None:
        return {}, decode_steps
    return step_updater.finish(), decode_steps


def main():
    parser = argparse.ArgumentParser()
    parse_common_args(parser)
    parser.add_argument("--lengths", type=str, default="8192,12288,16384,20480,24576,28672,32768")
    parser.add_argument("--measure_decode_tokens", type=int, default=1)
    parser.add_argument("--warmup_per_length", type=int, default=1)
    parser.add_argument("--measure_repeats", type=int, default=3)
    parser.add_argument("--estimate_total_tokens", type=int, default=128)
    parser.add_argument("--disable_flash_attn", action="store_true", default=True)
    parser.add_argument("--enable_flash_attn", action="store_true")
    parser.add_argument("--save_path", type=str, default="eval/LongBench2/results/quest_timing_stats.json")
    args = parser.parse_args()

    if args.enable_flash_attn:
        args.disable_flash_attn = False
    if args.disable_flash_attn:
        os.environ["DISABLE_FLASH_ATTN"] = "1"
    else:
        os.environ["DISABLE_FLASH_ATTN"] = "0"

    if args.model is None:
        raise ValueError("Please set --model")
    if args.method != "quest":
        raise ValueError("This script is intended for --method quest.")

    target_lengths = parse_lengths(args.lengths)
    target_lengths = sorted(target_lengths)
    if len(target_lengths) == 0:
        raise ValueError("No valid target lengths.")

    model_name = args.model
    model, tokenizer, step_updater, _, _ = load_model_and_tokenizer(
        model_map[model_name], args
    )

    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    item = dataset[int(args.data_idx or 0)]
    item = {
        "_id": item["_id"],
        "question": item["question"],
        "choice_A": item["choice_A"],
        "choice_B": item["choice_B"],
        "choice_C": item["choice_C"],
        "choice_D": item["choice_D"],
        "context": item["context"],
    }

    rows = []
    for target_len in target_lengths:
        context, actual_input_len = truncate_context_to_target_len(
            item, target_len, model_name, tokenizer
        )
        prompt = build_prompt(item, context)

        for _ in range(max(0, args.warmup_per_length)):
            _ = run_one_case(
                prompt,
                model_name,
                model,
                tokenizer,
                step_updater,
                decode_tokens=max(1, args.measure_decode_tokens),
            )

        per_token_select = []
        per_token_score = []
        per_token_topk = []
        per_token_pack = []
        per_token_attn = []
        measured_decode_steps = []

        for _ in range(max(1, args.measure_repeats)):
            stats, decode_steps = run_one_case(
                prompt,
                model_name,
                model,
                tokenizer,
                step_updater,
                decode_tokens=max(1, args.measure_decode_tokens),
            )
            measured_decode_steps.append(int(decode_steps))
            if decode_steps <= 0:
                per_token_select.append(0.0)
                per_token_score.append(0.0)
                per_token_topk.append(0.0)
                per_token_pack.append(0.0)
                per_token_attn.append(0.0)
                continue
            denom = float(decode_steps)
            per_token_select.append(float(stats.get("token_select_ms", 0.0)) / denom)
            per_token_score.append(float(stats.get("token_select_score_ms", 0.0)) / denom)
            per_token_topk.append(float(stats.get("token_select_topk_ms", 0.0)) / denom)
            per_token_pack.append(float(stats.get("token_pack_ms", 0.0)) / denom)
            per_token_attn.append(float(stats.get("attn_compute_ms", 0.0)) / denom)

        token_select_ms_per_token = sum(per_token_select) / float(len(per_token_select))
        token_select_score_ms_per_token = sum(per_token_score) / float(len(per_token_score))
        token_select_topk_ms_per_token = sum(per_token_topk) / float(len(per_token_topk))
        token_pack_ms_per_token = sum(per_token_pack) / float(len(per_token_pack))
        attn_compute_ms_per_token = sum(per_token_attn) / float(len(per_token_attn))

        # Keep an interpretable "measured total" as avg(per-token)*avg(decoded tokens).
        avg_decode_steps = (
            sum(measured_decode_steps) / float(len(measured_decode_steps))
            if len(measured_decode_steps) > 0
            else 0.0
        )
        token_select_ms_total = token_select_ms_per_token * avg_decode_steps
        token_select_score_ms_total = token_select_score_ms_per_token * avg_decode_steps
        token_select_topk_ms_total = token_select_topk_ms_per_token * avg_decode_steps
        token_pack_ms_total = token_pack_ms_per_token * avg_decode_steps
        attn_compute_ms_total = attn_compute_ms_per_token * avg_decode_steps

        est_select_ms = token_select_ms_per_token * float(args.estimate_total_tokens)
        est_select_score_ms = token_select_score_ms_per_token * float(args.estimate_total_tokens)
        est_select_topk_ms = token_select_topk_ms_per_token * float(args.estimate_total_tokens)
        est_pack_ms = token_pack_ms_per_token * float(args.estimate_total_tokens)
        est_attn_ms = attn_compute_ms_per_token * float(args.estimate_total_tokens)
        ratio = (
            token_select_ms_per_token / attn_compute_ms_per_token
            if attn_compute_ms_per_token > 0
            else None
        )
        row = {
            "target_input_len": int(target_len),
            "actual_input_len": int(actual_input_len),
            "decode_tokens_measured": int(round(avg_decode_steps)),
            "warmup_per_length": int(max(0, args.warmup_per_length)),
            "measure_repeats": int(max(1, args.measure_repeats)),
            "token_select_ms_total_measured": token_select_ms_total,
            "token_select_score_ms_total_measured": token_select_score_ms_total,
            "token_select_topk_ms_total_measured": token_select_topk_ms_total,
            "token_pack_ms_total_measured": token_pack_ms_total,
            "attn_compute_ms_total_measured": attn_compute_ms_total,
            "token_select_ms_per_token": token_select_ms_per_token,
            "token_select_score_ms_per_token": token_select_score_ms_per_token,
            "token_select_topk_ms_per_token": token_select_topk_ms_per_token,
            "token_pack_ms_per_token": token_pack_ms_per_token,
            "attn_compute_ms_per_token": attn_compute_ms_per_token,
            "token_select_ms_est_total": est_select_ms,
            "token_select_score_ms_est_total": est_select_score_ms,
            "token_select_topk_ms_est_total": est_select_topk_ms,
            "token_pack_ms_est_total": est_pack_ms,
            "attn_compute_ms_est_total": est_attn_ms,
            "estimate_total_tokens": int(args.estimate_total_tokens),
            "select_calls": int(stats.get("token_select_calls", 0)),
            "select_score_calls": int(stats.get("token_select_score_calls", 0)),
            "select_topk_calls": int(stats.get("token_select_topk_calls", 0)),
            "pack_calls": int(stats.get("token_pack_calls", 0)),
            "attn_calls": int(stats.get("attn_compute_calls", 0)),
            "select_over_attn_ratio": ratio,
        }
        rows.append(row)
        ratio_text = f"{ratio:.3f}" if ratio is not None else "NA"
        print(
            f"len={row['actual_input_len']}, "
            f"select_per_token_ms={row['token_select_ms_per_token']:.3f}, "
            f"score_per_token_ms={row['token_select_score_ms_per_token']:.3f}, "
            f"topk_per_token_ms={row['token_select_topk_ms_per_token']:.3f}, "
            f"pack_per_token_ms={row['token_pack_ms_per_token']:.3f}, "
            f"attn_per_token_ms={row['attn_compute_ms_per_token']:.3f}, "
            f"select_x{args.estimate_total_tokens}={row['token_select_ms_est_total']:.3f}, "
            f"score_x{args.estimate_total_tokens}={row['token_select_score_ms_est_total']:.3f}, "
            f"topk_x{args.estimate_total_tokens}={row['token_select_topk_ms_est_total']:.3f}, "
            f"pack_x{args.estimate_total_tokens}={row['token_pack_ms_est_total']:.3f}, "
            f"attn_x{args.estimate_total_tokens}={row['attn_compute_ms_est_total']:.3f}, "
            f"ratio={ratio_text}"
        )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved timing stats to: {args.save_path}")


if __name__ == "__main__":
    main()
