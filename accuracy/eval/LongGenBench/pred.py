import argparse
import time
import json
import torch
import os
from eval.util import parse_common_args, build_chat, load_model_and_tokenizer, get_out_path, sample_token
from tqdm import tqdm
RED = "\033[91m"  # Bright Red
RESET = "\033[0m" # Reset color

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'config')
model_map = json.loads(open(os.path.join(_CONFIG_DIR, 'model2path.json'), encoding='utf-8').read())
maxlen_map = json.loads(open(os.path.join(_CONFIG_DIR, 'model2maxlen.json'), encoding='utf-8').read())
stop_sign = '*** finished'

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM with command line arguments.')
    parse_common_args(parser)
    parser.add_argument('--input_type', type=str, default="short", choices=["short", "long"])
  
    args = parser.parse_args()
    return args

# Process output to split blocks and count words
def process_output(output: str) -> dict:
    blocks = output.split('#*#')
    word_count = len(output.split())
    return {"blocks": blocks, "word_count": word_count}

def read_json(file_path):
    with open(file_path, 'r',encoding='utf-8') as file:
        return json.load(file)
    
def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def save_to_json(data: list, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_inputs(filename: str) -> list:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def append_to_json(file_path, data_to_append):
    """
    Reads a JSON file expected to contain a list, appends a new object
    to that list, and writes the updated list back to the file.

    Args:
        file_path (str): The path to the JSON file.
        data_to_append (dict): The dictionary (JSON object) to append to the list.
    """
    json_list = []

    try:
        with open(file_path, 'r') as f:
            content = f.read()
            if content:
                json_list = json.loads(content)
                if not isinstance(json_list, list):
                    print(f"Error: File {file_path} does not contain a JSON list.")
                    json_list = []
    except FileNotFoundError:
        print(f"File {file_path} not found. A new file will be created.")
    except json.JSONDecodeError:
        assert False

    json_list.append(data_to_append)
    try:
        with open(file_path, 'w') as f:
            json.dump(json_list, f, indent=4)
    except IOError as e:
        assert False
    except TypeError as e:
        assert False

# Combine inputs, results and word counts and save them
def process_and_save_results(inputs: list, results: list, filename: str) -> None:
    combined = []
    for input_data, result_data in zip(inputs, results):
        combined.append({
            "input": input_data["prompt"],
            "checks_once": input_data["checks_once"],
            "checks_range": input_data["checks_range"],
            "checks_periodic": input_data["checks_periodic"],
            "type": input_data["type"],
            "number": input_data['number'],
            "output_blocks": result_data["blocks"],
            "word_count": result_data["word_count"]  # Adding word count here
        })
    save_to_json(combined, filename)


def get_pred(inputs, prompts, model_name, model, tokenizer, step_updater, eos_token_ids, 
             max_gen, temperature, top_p, stop_tokens, filename):
    max_len = maxlen_map[model_name]
    results = []
    for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        input_data = inputs[idx]
        input_ids = tokenizer.encode(prompt)
        assert len(input_ids) < max_len
        chat_prompt = build_chat(tokenizer, prompt, model_name)
        if isinstance(chat_prompt, str):
            input = tokenizer(chat_prompt, truncation=False, return_tensors="pt").to(
                "cuda"
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
            for _ in range(max_gen - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values
                pred_token_idx = sample_token(outputs, temperature, top_p)
                pred_token_idx_i = pred_token_idx.item()
                generated_content += [pred_token_idx_i]
                if step_updater is not None:
                    step_updater.update(pred_token_idx_i)
                if pred_token_idx_i in eos_token_ids:
                    break
                if generated_content[-len(stop_tokens):] == stop_tokens:
                    # print("stop!")
                    break
        output = tokenizer.decode(generated_content, skip_special_tokens=True)
        results.append(output)

        result_data = process_output(input_data['prefix']+ output)
        to_save = {
            "input": input_data["prompt"],
            "checks_once": input_data["checks_once"],
            "checks_range": input_data["checks_range"],
            "checks_periodic": input_data["checks_periodic"],
            "type": input_data["type"],
            "number": input_data['number'],
            "output_blocks": result_data["blocks"],
            "word_count": result_data["word_count"],    # Adding word count here
            "output_len": len(generated_content),
        }
        if step_updater is not None:
            to_save.update(step_updater.finish())
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        append_to_json(filename, to_save)
    return results
        

def get_pred_batched(prompts, model_name, model, tokenizer, step_updater, eos_token_ids, 
                     max_gen, temperature, top_p, stop_tokens, batch_size):
    device = next(model.parameters()).device
    max_len = maxlen_map[model_name]
    results = []
    tokenizer.pad_token = tokenizer.eos_token
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompt = prompts[i: i+batch_size]
        current_batch_size = len(batch_prompt)
        assert current_batch_size > 0
        batch_chat_prompts = [
            build_chat(tokenizer, p_text, model_name, to_token=False) for p_text in batch_prompt
        ]
        # print(batch_chat_prompts)

        inputs = tokenizer(
            batch_chat_prompts,
            return_tensors="pt",
            padding=True, # Pad to the longest sequence in the batch
            truncation=False,
        ).to(device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask  # padding mask
        assert len(input_ids[0]) < max_len
        print(type(input_ids), input_ids.shape)
        input_ids = input_ids.repeat(args.repeat_bsz, 1)
        attention_mask = attention_mask.repeat(args.repeat_bsz, 1)
        current_batch_size = args.repeat_bsz
        print(type(input_ids), input_ids.shape)

        # Initialize for generation
        generated_tokens_batch = [[] for _ in range(current_batch_size)]
        # Keep track of active sequences (not yet finished by EOS)
        active_sequences = torch.ones(current_batch_size, dtype=torch.bool, device=device)
        
        # Store past_key_values
        past_key_values = None

        st = time.perf_counter()
        with torch.no_grad():
            if step_updater is not None:
                step_updater.reset(input_ids)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            next_tokens = sample_token(output, temperature, top_p)
            for k in range(current_batch_size):
                if active_sequences[k]:
                    token_int = next_tokens[k, 0].item()
                    generated_tokens_batch[k].append(token_int)
                    if token_int in eos_token_ids:
                        active_sequences[k] = False
            if step_updater is not None:
                step_updater.update(next_tokens)


            for _ in range(max_gen - 1):
                if not torch.any(active_sequences): # Stop if all sequences are done
                    break

                current_input_ids = next_tokens
                effective_attention_mask = active_sequences.unsqueeze(-1).expand_as(current_input_ids)
                attention_mask = torch.cat([attention_mask, effective_attention_mask], dim=-1)
                outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_tokens = sample_token(outputs, temperature, top_p)

                for k in range(current_batch_size):
                    if active_sequences[k]:
                        token_int = next_tokens[k, 0].item()
                        generated_tokens_batch[k].append(token_int)
                        if token_int in eos_token_ids:
                            active_sequences[k] = False
                if step_updater is not None:
                    step_updater.update(next_tokens)
        ed = time.perf_counter()
        gen_token = [len(o) for o in generated_tokens_batch]
        print(f"{RED} {model_name} {args.method} Generate {gen_token} tokens {RESET}")
        total_time = ed - st
        print(f"{RED} Total time: {total_time:.2f}s {RESET}")

        for k in range(current_batch_size):
            pred_text = tokenizer.decode(generated_tokens_batch[k], skip_special_tokens=True)
            results.append(pred_text)

    return results

if __name__ == "__main__":
    args = parse_args()

    input_file = f"eval/LongGenBench/Dataset_{args.input_type}.json"
    inputs = load_inputs(input_file)

    prompts = [input_data['prompt'] for input_data in inputs]
    if args.data_idx is not None:
        prompts = [prompts[args.data_idx]]
    elif args.data_from is not None:
        assert args.data_idx_to is not None
        prompts = prompts[args.data_from: args.data_idx_to]
    elif args.data_idx_to is not None:
        prompts = prompts[:args.data_idx_to]

    model_name = args.model
    model, tokenizer, step_updater, eos_token_ids, config = load_model_and_tokenizer(
        model_map[model_name], args
    )
    stop_tokens = tokenizer(stop_sign, truncation=False).input_ids

    out_root_dir = f"eval/LongGenBench/results_{args.input_type}"
    out_path = get_out_path(args, config, out_root_dir, False)
    head, tail = os.path.split(out_path)
    parent_dir, last_component = os.path.split(head)
    new_filename = last_component + tail
    out_path = os.path.join(parent_dir, new_filename)
    print(f"\nSaved result to {out_path}")

    if args.data_idx is not None and args.data_idx_to is not None:
        if args.input_type == "short":
            assert args.max_gen == 16000
        else:
            assert args.max_gen == 32000

    if args.batch_size == 1 and args.repeat_bsz is None:
        outputs = get_pred(inputs, prompts, model_name, model, tokenizer, step_updater, eos_token_ids, 
                           args.max_gen, args.temperature, args.top_p, stop_tokens, out_path)
    else:
        outputs = get_pred_batched(
            prompts, model_name, model, tokenizer, step_updater, eos_token_ids, 
            args.max_gen, args.temperature, args.top_p, stop_tokens, args.batch_size
        )
        
        results = [process_output(input['prefix']+ output) for output, input in zip(outputs,inputs)]
        process_and_save_results(inputs, results, out_path)