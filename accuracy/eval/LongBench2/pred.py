import os, csv, json
import argparse
from tqdm import tqdm
from datasets import load_dataset
import re
import torch
from eval.util import parse_common_args, build_chat, load_model_and_tokenizer, get_out_path

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'config')
model_map = json.loads(open(os.path.join(_CONFIG_DIR, 'model2path.json'), encoding='utf-8').read())
maxlen_map = json.loads(open(os.path.join(_CONFIG_DIR, 'model2maxlen.json'), encoding='utf-8').read())

template_rag = open('eval/LongBench2/prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('eval/LongBench2/prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('eval/LongBench2/prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('eval/LongBench2/prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('eval/LongBench2/prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def query_llm(prompt, model_name, model, 
              tokenizer, step_updater, eos_token_ids,
              temperature=0.5, max_new_tokens=128):
    # truncate
    max_len = maxlen_map[model_name]
    if model_name in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        assert False

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
        # print(input.shape)
        output = model(
            input_ids=input,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_content = [pred_token_idx.item()]
        if step_updater is not None:
            step_updater.update(pred_token_idx.item())
        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content += [pred_token_idx.item()]
            if step_updater is not None:
                step_updater.update(pred_token_idx.item())
            if pred_token_idx.item() in eos_token_ids:
                break
    
    return tokenizer.decode(generated_content, skip_special_tokens=True), len(generated_content)

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred(model_name, model, tokenizer, 
             step_updater, eos_token_ids,
             data, args, fout):
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        if args.cot:
            output, out_len = query_llm(prompt, model_name, model, 
                               tokenizer, step_updater, eos_token_ids,
                               temperature=0.1, max_new_tokens=1024)
        else:
            output, out_len = query_llm(prompt, model_name, model, 
                               tokenizer, step_updater, eos_token_ids,
                               temperature=0.1, max_new_tokens=128)
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output, out_len = query_llm(prompt, model_name, model, 
                               tokenizer, step_updater, eos_token_ids,
                               temperature=0.1, max_new_tokens=128)
            if output == '':
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        item['output_len'] = out_len
        item.update(step_updater.finish())
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    model_name = args.model
    model, tokenizer, step_updater, eos_token_ids, config = load_model_and_tokenizer(
        model_map[model_name], args
    )
    out_root_dir = "eval/LongBench2/results"
    out_path = get_out_path(args, config, out_root_dir, False)
    head, tail = os.path.split(out_path)
    parent_dir, last_component = os.path.split(head)
    new_filename = last_component + tail
    out_path = os.path.join(parent_dir, new_filename)
    print(args)
    if args.rag > 0:
        out_file = out_path.replace(".jsonl", f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = out_path.replace(".jsonl", "_no_context.jsonl")
    elif args.cot:
        out_file = out_path.replace(".jsonl", "_cot.jsonl")
    else:
        out_file = out_path
    print("Outfile:", out_file)
    os.makedirs(os.path.split(out_file)[0], exist_ok=True)

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    if args.data_idx is not None:
        dataset = dataset.select(range(args.data_idx, args.data_idx+1))
    elif args.data_idx_to is not None:
        dataset = dataset.select(range(0, args.data_idx_to))
    elif args.data_from is not None:
        dataset = dataset.select(range(args.data_from, len(data)))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    get_pred(model_name, model, tokenizer, step_updater, eos_token_ids, data, args, fout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parse_common_args(parser)
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    args = parser.parse_args()
    main()