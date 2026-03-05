# Accuracy Evaluation

This directory contains evaluation pipelines for **LongBench2**, **LongGenBench**, and **Reasoning** tasks. All commands should be run from the `accuracy/` directory.

```bash
cd accuracy
```
For razor attention patterns:
```bash
tar -xzvf razor_patterns.tar.gz
```

## Methods

| Method | `--method` flag | Description |
|---|---|---|
| Full attention | `full` | No KV cache eviction (baseline) |
| Razor | `razor` | Razor-Attention with a sparsity pattern file |
| Quest | `quest` | Quest page-level retrieval |
| ArkVale | `arkv` | ArkVale retrieval |
| RaaS | `raas` | Retrieval-as-a-Service |
| **FreeKV** (ours) | `spec_ret` | Speculative retrieval + correction |

## Common Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | — | Model name (key in `config/model2path.json`) |
| `--method` | `full` | KV cache method (see table above) |
| `--budget` | — | Total KV cache token budget |
| `--sink` | — | Number of sink tokens |
| `--recent` | — | Number of recent tokens |
| `--skip_layer` | 1 | Number of layers to skip from sparse attention |
| `--sparsity` | 1 | Sparsity level |
| `--max_gen` | 8192 | Max new tokens to generate |
| `--temperature` | 0.0 | Sampling temperature (0 = greedy) |
| `--top_p` | 1.0 | Top-p (nucleus) sampling threshold |
| `--seed` | 42 | Random seed |
| `--out_root_dir` | — | Root directory for output files |

FreeKV-specific:

| Argument | Description |
|---|---|
| `--page_rep` | Page representation (`quest` or `arkv`) |
| `--spec_ret_steps` | Speculative retrieval steps |
| `--spec_ret_corr` | Correction cosine-similarity threshold |

---

## LongBench2

### Predict

**Full attention:**
```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.LongBench2.pred \
  --model qwen-2.5-chat-7b --method full \
  --out_root_dir eval/LongBench2/res
```

**FreeKV (ours):**
```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.LongBench2.pred \
  --model qwen-2.5-chat-7b --method spec_ret \
  --page_rep quest --GQA_policy avgSM --spec_ret_steps 2 \
  --spec_ret_corr 0.8 \
  --sink 128 --recent 128 --budget 1792 \
  --out_root_dir eval/LongBench2/res
```

**Other baselines:**
```bash
# Razor
CUDA_VISIBLE_DEVICES=0 python -m eval.LongBench2.pred \
  --model qwen-2.5-chat-7b --method razor \
  --threshold 0.5 --attn_load_dir razor_patterns/qwen-2.5-chat-7b/<pattern.tsv> \
  --sink 128 --recent 128 \
  --out_root_dir eval/LongBench2/res

# Quest
CUDA_VISIBLE_DEVICES=0 python -m eval.LongBench2.pred \
  --model qwen-2.5-chat-7b --method quest \
  --GQA_policy maxS \
  --sink 128 --recent 128 --budget 1792 \
  --out_root_dir eval/LongBench2/res

# ArkVale
CUDA_VISIBLE_DEVICES=0 python -m eval.LongBench2.pred \
  --model qwen-2.5-chat-7b --method arkv \
  --sink 128 --recent 128 --budget 1792 \
  --out_root_dir eval/LongBench2/res

# RaaS
CUDA_VISIBLE_DEVICES=0 python -m eval.LongBench2.pred \
  --model qwen-2.5-chat-7b --method raas \
  --sink 128 --recent 128 --budget 1792 \
  --out_root_dir eval/LongBench2/res
```

### Evaluate

```bash
python eval/LongBench2/result.py
```

Results are written to `eval/LongBench2/results/result.csv`.

---

## LongGenBench

### Predict

**FreeKV (ours):**
```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.LongGenBench.pred \
  --model qwen-2.5-chat-7b --method spec_ret \
  --page_rep quest --GQA_policy avgSM --spec_ret_steps 2 \
  --spec_ret_corr 0.9 \
  --max_gen 16000 --temperature 0.95 \
  --out_root_dir eval/LongGenBench/res
```

**Other baselines:**
```bash
# Razor
CUDA_VISIBLE_DEVICES=0 python -m eval.LongGenBench.pred \
  --model qwen-2.5-chat-7b --method razor \
  --threshold 0.5 --attn_load_dir razor_patterns/qwen-2.5-chat-7b/<pattern.tsv> \
  --max_gen 16000 --temperature 0.95 \
  --out_root_dir eval/LongGenBench/res

# Quest
CUDA_VISIBLE_DEVICES=0 python -m eval.LongGenBench.pred \
  --model qwen-2.5-chat-7b --method quest \
  --GQA_policy maxS \
  --max_gen 16000 --temperature 0.95 \
  --out_root_dir eval/LongGenBench/res

# ArkVale
CUDA_VISIBLE_DEVICES=0 python -m eval.LongGenBench.pred \
  --model qwen-2.5-chat-7b --method arkv \
  --max_gen 16000 --temperature 0.95 \
  --out_root_dir eval/LongGenBench/res

# RaaS
CUDA_VISIBLE_DEVICES=0 python -m eval.LongGenBench.pred \
  --model qwen-2.5-chat-7b --method raas \
  --max_gen 16000 --temperature 0.95 \
  --out_root_dir eval/LongGenBench/res
```

### Evaluate

```bash
python eval/LongGenBench/eval.py --data <prediction_json> --csv <output_csv>
```
(vllm is required for evaluate LongGenBench using local models)

---

## Reasoning (AIME24 / GPQA50 / MATH500)

Reasoning tasks use sampling with `--temperature 0.6 --top_p 0.95` and run multiple seeds for robust evaluation.

### Predict

**FreeKV (ours):**
```bash
for seed in {42..49}; do
  CUDA_VISIBLE_DEVICES=0 python -u -m eval.reasoning.pred \
    --model ds-r1-qwen-7b --dataset AIME24 \
    --method spec_ret --page_rep quest --GQA_policy avgSM \
    --spec_ret_steps 2 \
    --spec_ret_corr 0.9 \
    --temperature 0.6 --top_p 0.95 --max_gen 16384 \
    --seed $seed --out_root_dir eval/reasoning/res
done
```

**Other baselines:**
```bash
# Razor
for seed in {42..49}; do
  CUDA_VISIBLE_DEVICES=0 python -u -m eval.reasoning.pred \
    --model ds-r1-qwen-7b --dataset AIME24 \
    --method razor --threshold 0.5 --attn_load_dir razor_patterns/ds-r1-qwen-7b/<pattern.tsv> \
    --temperature 0.6 --top_p 0.95 --max_gen 16384 \
    --seed $seed \
    --out_root_dir eval/reasoning/res
done

# Quest
for seed in {42..49}; do
  CUDA_VISIBLE_DEVICES=0 python -u -m eval.reasoning.pred \
    --model ds-r1-qwen-7b --dataset AIME24 \
    --method quest --GQA_policy maxS \
    --temperature 0.6 --top_p 0.95 --max_gen 16384 \
    --seed $seed \
    --out_root_dir eval/reasoning/res
done

# ArkVale
for seed in {42..49}; do
  CUDA_VISIBLE_DEVICES=0 python -u -m eval.reasoning.pred \
    --model ds-r1-qwen-14b --dataset AIME24 \
    --method arkv \
    --temperature 0.6 --top_p 0.95 --max_gen 16384 \
    --seed $seed \
    --out_root_dir eval/reasoning/res
done

# RaaS
for seed in {42..49}; do
  CUDA_VISIBLE_DEVICES=0 python -u -m eval.reasoning.pred \
    --model ds-r1-llama-8b --dataset AIME24 \
    --method raas \
    --temperature 0.6 --top_p 0.95 --max_gen 16384 \
    --seed $seed \
    --out_root_dir eval/reasoning/res
done
```

Available datasets: `AIME24`, `GPQA50`, `MATH500` (see `eval/reasoning/datasets/`).

### Evaluate

```bash
python eval/reasoning/eval.py --data_dir <results_dir> --dataset <dataset_name>
```

