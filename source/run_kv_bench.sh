#!/usr/bin/env bash
set -euo pipefail

# Simple benchmark runner:
# - models: llama3.1-8b-chat-8k, qwen-2.5-7b
# - tasks: gov_report, lgbench
# - modes:
#   - freekv: recall_impl=cuda_cpy + spec_ret + corr=0.9
#   - echokv: sel_policy=echokv_token + flash_attn split_overlap
# - bsz: 1,2,4
# - repeats: gov_report=3, lgbench=2
#
# Usage:
#   bash source/run_kv_bench.sh
# Optional env:
#   CUDA_VISIBLE_DEVICES=1
#   DATA_IDX=0
#   WARMUP=1
#   PYTHON_BIN=python

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-1}"
DATA_IDX="${DATA_IDX:-0}"
WARMUP="${WARMUP:-1}"

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
  local mode="$5"   # freekv or echokv
  local run_idx="$6"
  local total_runs="$7"

  local base_cmd=(
    "$PYTHON_BIN" source/pred.py
    --model "$model"
    --dataset "$task"
    --data_idx "$DATA_IDX"
    --max_gen "$max_gen"
    --warmup "$WARMUP"
    --repeat_bsz "$bsz"
  )

  echo "[$run_idx/$total_runs] model=$model task=$task mode=$mode bsz=$bsz"

  if [[ "$mode" == "echokv" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" "${base_cmd[@]}" \
      --sel_policy echokv_token \
      --echo_flash_mode split_overlap \
      --echo_stream_chunk_pages 32 \
      --echo_attn_backend flash_attn \
      --echo_shared_batch
  else
    CUDA_VISIBLE_DEVICES="$GPU_ID" "${base_cmd[@]}" \
      --recall_impl cuda_cpy \
      --spec_ret \
      --corr 0.9
  fi
}

TOTAL_RUNS=0
for _m in "${MODELS[@]}"; do
  for _b in "${BSZS[@]}"; do
    TOTAL_RUNS=$((TOTAL_RUNS + 3 * 2)) # gov_report: (freekv+echokv)*3
    TOTAL_RUNS=$((TOTAL_RUNS + 2 * 2)) # lgbench:    (freekv+echokv)*2
  done
done

CUR=0
for model in "${MODELS[@]}"; do
  for bsz in "${BSZS[@]}"; do
    # gov_report: max_gen=512, repeat=3
    for mode in freekv echokv; do
      for r in 1 2 3; do
        CUR=$((CUR + 1))
        run_cmd "$model" "gov_report" "512" "$bsz" "$mode" "$CUR" "$TOTAL_RUNS"
      done
    done

    # lgbench: max_gen=16384, repeat=2
    for mode in freekv echokv; do
      for r in 1 2; do
        CUR=$((CUR + 1))
        run_cmd "$model" "lgbench" "16384" "$bsz" "$mode" "$CUR" "$TOTAL_RUNS"
      done
    done
  done
done

echo "All runs completed: $CUR/$TOTAL_RUNS"
