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
    eos_token_ids,
    max_new_tokens: int,
) -> dict:
    input_ids = build_input_ids(prompt, model_name, tokenizer)
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
        if step_updater is not None:
            step_updater.update(pred_token_idx.item())
        for _ in range(max_new_tokens - 1):
            output = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            if step_updater is not None:
                step_updater.update(pred_token_idx.item())
            if pred_token_idx.item() in eos_token_ids:
                break
    if step_updater is None:
        return {}
    return step_updater.finish()


def main():
    parser = argparse.ArgumentParser()
    parse_common_args(parser)
    parser.add_argument("--lengths", type=str, default="8192,12288,16384,20480,24576,28672,32768")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="eval/LongBench2/results/quest_timing_stats.json")
    args = parser.parse_args()

    if args.model is None:
        raise ValueError("Please set --model")
    if args.method != "quest":
        raise ValueError("This script is intended for --method quest.")

    target_lengths = parse_lengths(args.lengths)
    target_lengths = sorted(target_lengths)
    if len(target_lengths) == 0:
        raise ValueError("No valid target lengths.")

    model_name = args.model
    model, tokenizer, step_updater, eos_token_ids, _ = load_model_and_tokenizer(
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

    # Warmup one short run to reduce first-kernel compilation noise.
    warm_context, _ = truncate_context_to_target_len(
        item, target_lengths[0], model_name, tokenizer
    )
    warm_prompt = build_prompt(item, warm_context)
    _ = run_one_case(
        warm_prompt,
        model_name,
        model,
        tokenizer,
        step_updater,
        eos_token_ids,
        max_new_tokens=1,
    )

    rows = []
    for target_len in target_lengths:
        context, actual_input_len = truncate_context_to_target_len(
            item, target_len, model_name, tokenizer
        )
        prompt = build_prompt(item, context)
        stats = run_one_case(
            prompt,
            model_name,
            model,
            tokenizer,
            step_updater,
            eos_token_ids,
            args.max_new_tokens,
        )
        token_select_ms = float(stats.get("token_select_ms", 0.0))
        attn_compute_ms = float(stats.get("attn_compute_ms", 0.0))
        ratio = (token_select_ms / attn_compute_ms) if attn_compute_ms > 0 else None
        row = {
            "target_input_len": int(target_len),
            "actual_input_len": int(actual_input_len),
            "token_select_ms": token_select_ms,
            "attn_compute_ms": attn_compute_ms,
            "select_calls": int(stats.get("token_select_calls", 0)),
            "attn_calls": int(stats.get("attn_compute_calls", 0)),
            "select_over_attn_ratio": ratio,
        }
        rows.append(row)
        ratio_text = f"{ratio:.3f}" if ratio is not None else "NA"
        print(
            f"len={row['actual_input_len']}, "
            f"select_ms={row['token_select_ms']:.3f}, "
            f"attn_ms={row['attn_compute_ms']:.3f}, "
            f"ratio={ratio_text}"
        )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved timing stats to: {args.save_path}")


if __name__ == "__main__":
    main()
