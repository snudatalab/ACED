#!/bin/bash
# Example: compress Qwen3-30B-A3B with ACED end to end.
# Requires the base checkpoint to already be downloaded locally and the
# calibration dataset to be reachable (see README "Before you start").
set -e

cd "$(dirname "$0")"

MODEL="qwen3"
OUT="results/${MODEL}_example"

echo "=== [1/2] src/aced.py -- I1 orientation search + I2 basis-count search (+ I3 config) ==="
python src/aced.py \
  --model "$MODEL" \
  --device cuda \
  --activation tanh \
  --target_compression_ratio 0.5 \
  --final_epochs 3000 \
  --lr 0.07 \
  --I2_num_candidates 8 \
  --I2_search_ratio 0.2 \
  --I2_num_search_layers 5 \
  --I3_use_pcc \
  --I3_calib_data_name pile \
  --I3_calib_num_samples 64 \
  --I3_calib_seqlen 512 \
  --I3_lambda_pcc 0.001 \
  --I3_tokens_batch 128 \
  --aced_result_path "$OUT/aced_result.json"

echo ""
echo "=== [2/2] src/train.py -- final decomposition training + save ==="
# All I3_* settings above are carried over automatically via aced_result.json.
# Pass any --I3_* flag here again to override just that one setting for this run.
python src/train.py \
  --aced_result_path "$OUT/aced_result.json" \
  --device cuda \
  --save_model \
  --save_format both \
  --trust_remote_code
