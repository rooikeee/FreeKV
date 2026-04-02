#!/usr/bin/env bash
set -euo pipefail

# ArkVale benchmark runner:
# - models: llama-3.1-chat-8b, qwen-2.5-chat-7b
# - tasks: gov_report, lgbench (long decoding)
# - mode: sel_policy=topk + recall_impl=arkvale
# - bsz: 1,2,4
# - repeats: gov_report=3, lgbench=2
#
# Usage:
#   bash source/run_kv_bench_arkvale.sh
#
# Optional env:
#   CUDA_VISIBLE_DEVICES=1
#   DATA_IDX=0
#   WARMUP=1
#   PYTHON_BIN=python
#   GOV_REPEAT=3
#   LGBENCH_REPEAT=2

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-1}"
DATA_IDX="${DATA_IDX:-0}"
WARMUP="${WARMUP:-1}"
GOV_REPEAT="${GOV_REPEAT:-3}"
LGBENCH_REPEAT="${LGBENCH_REPEAT:-2}"

# Keep model keys aligned with config/model2path.json.
MODELS=(
  "llama-3.1-chat-8b"
  "qwen-2.5-chat-7b"
)

BSZS=(1 2 4)

run_cmd() {
  local model="$1"
  local task="$2"
  local max_gen="$3"
  local bsz="$4"
  local run_idx="$5"
  local total_runs="$6"

  echo "[$run_idx/$total_runs] model=$model task=$task mode=arkvale bsz=$bsz"

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" source/pred.py \
    --model "$model" \
    --dataset "$task" \
    --data_idx "$DATA_IDX" \
    --max_gen "$max_gen" \
    --warmup "$WARMUP" \
    --repeat_bsz "$bsz" \
    --sel_policy topk \
    --recall_impl arkvale
}

TOTAL_RUNS=0
for _m in "${MODELS[@]}"; do
  for _b in "${BSZS[@]}"; do
    TOTAL_RUNS=$((TOTAL_RUNS + GOV_REPEAT))
    TOTAL_RUNS=$((TOTAL_RUNS + LGBENCH_REPEAT))
  done
done

CUR=0
for model in "${MODELS[@]}"; do
  for bsz in "${BSZS[@]}"; do
    # gov_report: max_gen=512
    for ((r=1; r<=GOV_REPEAT; r++)); do
      CUR=$((CUR + 1))
      run_cmd "$model" "gov_report" "512" "$bsz" "$CUR" "$TOTAL_RUNS"
    done

    # lgbench: max_gen=16384 (long decoding)
    for ((r=1; r<=LGBENCH_REPEAT; r++)); do
      CUR=$((CUR + 1))
      run_cmd "$model" "lgbench" "16384" "$bsz" "$CUR" "$TOTAL_RUNS"
    done
  done
done

echo "All ArkVale runs completed: $CUR/$TOTAL_RUNS"

