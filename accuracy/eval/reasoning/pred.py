import os
import time
import argparse
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import random
from requests.exceptions import ProxyError, SSLError
from eval.util import parse_common_args, build_chat, load_model_and_tokenizer, get_out_path, sample_token
try:
    import torch_npu
    use_npu = True
except ImportError:
    use_npu = False
    pass

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_with_retry(model_path, args, retries=3, delay=1):
    for attempt in range(retries):
        try:
            model, tokenizer, step_updater, eos_token_ids, config = load_model_and_tokenizer(model_path, args)
            return model, tokenizer, step_updater, eos_token_ids, config
        except (ProxyError, SSLError) as e:
            print(f"Attempt {attempt + 1} failed due to network error: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                raise  # Re-raise the last exception if all retries fail


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parse_common_args(parser)
    return parser.parse_args(args)


def get_pred(
    model,
    tokenizer,
    eos_token_ids,
    data,
    answer_field_id,
    max_length,
    max_gen,
    prompt_format,
    model_name,
    temperature,
    top_p,
    step_updater,
    out_path
):
    preds = []
    for di, json_obj in enumerate(tqdm(data)):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            assert False
        
        # print(prompt)
        chat_prompt = build_chat(tokenizer, prompt, model_name)
        # print(chat_prompt)
        if isinstance(chat_prompt, str):
            input = tokenizer(chat_prompt, truncation=False, return_tensors="pt").to(
                "cuda" if not use_npu else "npu"
            ).input_ids
        else:
            input = chat_prompt
        with torch.no_grad():
            if step_updater is not None:
                step_updater.reset(input)
            output = model(
                input_ids=input,
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            pred_token_idx = sample_token(output, temperature, top_p)
            generated_content = [pred_token_idx.item()]
            if step_updater is not None:
                step_updater.update(pred_token_idx.item())

            # st = time.time()
            for gen_iter in range(max_gen - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                pred_token_idx = sample_token(outputs, temperature, top_p)
                pred_token_idx_int = pred_token_idx.item()
                generated_content += [pred_token_idx_int]
                if step_updater is not None:
                    step_updater.update(pred_token_idx_int)
                if pred_token_idx_int in eos_token_ids:
                    break
            # cost = time.time() - st
            # print(f"{max_gen} tokens use {cost:.2f}s, tbt {cost / max_gen * 1000:.2f} ms")
        stat = {}
        if step_updater is not None:
            stat.update(step_updater.finish())
        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        # print(pred)
        preds.append(
            {
                "qid": di + (args.data_from or 0),
                "input:": prompt,
                "pred": pred,
                "answer": json_obj[answer_field_id],
                "input_len": len(input[0]),
                "output_len": len(generated_content),
                **stat
            }
        )
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(preds[-1], f, ensure_ascii=False)
            f.write("\n")
    return preds


def get_pred_batched(
    model,
    tokenizer,
    eos_token_ids, # Can be a list of token IDs
    data,
    answer_field_id,
    max_gen,    # Max tokens to generate
    prompt_format,
    model_name,
    temperature,
    top_p,
    step_updater, # NOTE: step_updater logic will be complex with batching if not designed for it
    out_path,
    batch_size,
):
    device = next(model.parameters()).device
    preds_all = []
    # Ensure eos_token_ids is a tensor on the correct device for efficient checking
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch_json_objs = data.select(range(i, i+batch_size))
        current_batch_size = len(batch_json_objs)

        batch_prompts_text = [prompt_format.format(**json_obj) for json_obj in batch_json_objs]
        batch_chat_prompts_text = [
            build_chat(tokenizer, p_text, model_name) for p_text in batch_prompts_text
        ]
        
        inputs = tokenizer(
            batch_chat_prompts_text,
            return_tensors="pt",
            padding=True, # Pad to the longest sequence in the batch
            truncation=False,
        ).to(device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask  # padding mask
        
        assert input_ids.shape[1] > 0

        # Initialize for generation
        generated_tokens_batch = [[] for _ in range(current_batch_size)]
        # Keep track of active sequences (not yet finished by EOS)
        active_sequences = torch.ones(current_batch_size, dtype=torch.bool, device=device)
        
        # Store past_key_values
        past_key_values = None

        # Perform initial forward pass (prefill)
        with torch.no_grad():
            if current_batch_size > 0 : # Ensure there's something to process
                if step_updater:
                    step_updater.reset(input_ids) 
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_tokens = sample_token(outputs, temperature, top_p) # [batch_size, 1]

                # Add first generated token
                for k in range(current_batch_size):
                    if active_sequences[k]:
                        token_int = next_tokens[k, 0].item()
                        generated_tokens_batch[k].append(token_int)
                        if token_int in eos_token_ids:
                            active_sequences[k] = False
                if step_updater is not None:
                    step_updater.update(next_tokens)
                
        # Autoregressive generation loop
        for gen_iter in range(max_gen - 1):
            if not torch.any(active_sequences): # Stop if all sequences are done
                break

            with torch.no_grad():
                # [batch_size, 1]
                current_input_ids = next_tokens
                effective_attention_mask = active_sequences.unsqueeze(-1).expand_as(current_input_ids)
                attention_mask = torch.cat([attention_mask, effective_attention_mask], dim=-1)
                outputs = model(
                    input_ids=current_input_ids, # Should be shape [batch_size, 1]
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_tokens = sample_token(outputs, temperature, top_p)

                # Update generated tokens and active status
                for k in range(current_batch_size):
                    if active_sequences[k]:
                        token_int = next_tokens[k, 0].item()
                        generated_tokens_batch[k].append(token_int)
                        if token_int in eos_token_ids:
                            active_sequences[k] = False
                if step_updater is not None:
                    step_updater.update(next_tokens)
        
        # Decode and store predictions for the batch
        for k in range(current_batch_size):
            json_obj = batch_json_objs[k]
            original_prompt_text = batch_prompts_text[k] # Text before build_chat
            pred_text = tokenizer.decode(generated_tokens_batch[k], skip_special_tokens=True)
            
            stat = {}
            if step_updater is not None:
                stat.update(step_updater.finish())

            pred_entry = {
                "qid": i + k, # Global index
                "input:": original_prompt_text, # Original prompt before chat formatting
                "chat_input:": batch_chat_prompts_text[k], # What was actually tokenized as model input
                "pred": pred_text,
                "answer": json_obj[answer_field_id],
                "input_len": len(input_ids[k]), # Length of tokenized chat_input
                "output_len": len(generated_tokens_batch[k]),
                **stat
            }
            preds_all.append(pred_entry)
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(pred_entry, f, ensure_ascii=False)
                f.write("\n")
                
    return preds_all


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    _config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'config')
    model2path = json.load(open(os.path.join(_config_dir, 'model2path.json'), "r"))
    model2maxlen = json.load(open(os.path.join(_config_dir, 'model2maxlen.json'), "r"))
    device_list = ([i for i in range(torch.cuda.device_count())] if not use_npu else
                   [i for i in range(torch.npu.device_count())])
    model_name = args.model
    assert model_name in model2path and "Not allowed model"

    model, tokenizer, step_updater, eos_token_ids, config = load_model_with_retry(model2path[model_name], args)
    # do not use together with `device_map="auto"`
    # only enable_pp can be used for qwen models
    # model = to_device(model, device_list, enable_pp=True)

    max_length = model2maxlen[model_name]
    max_gen = args.max_gen

    dataset = args.dataset
    ds_dir = "eval/o1/datasets"
    answer_field_id = "answer"
    if dataset == "AIME":
        ds_path = f"{ds_dir}/aime.jsonl"
    elif dataset == "AIME24":
        ds_path = f"{ds_dir}/aime24.jsonl"
    elif dataset == "GPQAd":
        ds_path = f"{ds_dir}/gpqa_diamond.jsonl"
        answer_field_id = "Correct Answer"
    elif dataset == "GPQAm":
        ds_path = f"{ds_dir}/gpqa_main.jsonl"
        answer_field_id = "Correct Answer"
    elif dataset == "GPQA50":
        ds_path = f"{ds_dir}/gpqa50.jsonl"
        answer_field_id = "Correct Answer"
    elif dataset == "GPQA50c":
        ds_path = f"{ds_dir}/gpqa50c.jsonl"
    elif dataset == "MATH500":
        ds_path = f"{ds_dir}/math500.jsonl"
    elif dataset == "MATH50":
        ds_path = f"{ds_dir}/math50.jsonl"
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    data = load_dataset("json", data_files=ds_path, split="train")
    
    dataset2prompt = json.load(open("eval/o1/config/dataset2prompt.json", "r"))
    prompt_format = dataset2prompt[dataset]
    if "cot" in model_name:
        prompt_format += "<Thought> {thought} </Thought>\n"
    
    if args.data_idx is not None:
        data = data.select(range(args.data_idx, args.data_idx+1))
    elif args.data_idx_to is not None:
        data = data.select(range(0, args.data_idx_to))
    elif args.data_from is not None:
        data = data.select(range(args.data_from, len(data)))
    
    out_path = get_out_path(args, config)
    if args.batch_size == 1:
        preds = get_pred(
            model,
            tokenizer,
            eos_token_ids,
            data,
            answer_field_id,
            max_length,
            max_gen,
            prompt_format,
            model_name,
            args.temperature,
            args.top_p,
            step_updater,
            out_path
        )
    else:
        preds = get_pred_batched(
            model,
            tokenizer,
            eos_token_ids,
            data,
            answer_field_id,
            max_gen,
            prompt_format,
            model_name,
            args.temperature,
            args.top_p,
            step_updater,
            out_path,
            args.batch_size,
        )
