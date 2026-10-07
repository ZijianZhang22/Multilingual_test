#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

TRAIN_FRACTION="${TRAIN_FRACTION:-0.20}"
EVAL_MAX_BLOCKS="${EVAL_MAX_BLOCKS:-128}"
LAMBDA_PRESERVE="${LAMBDA_PRESERVE:-3.0}"
OUT_DIR="${OUT_DIR:-random_layer_freeze_runs/en_to_zh_seed0_lam${LAMBDA_PRESERVE}}"

python experiments/random_layer_freeze_pilot/run_pilot.py \
  --anchor_checkpoint invariance_runs/sequence_seed0/en__zh/stage1_en \
  --data_dir invariance_data/wiki \
  --old_language en \
  --new_language zh \
  --layers 12 20 24 \
  --freeze_ratios 0.2 0.8 1.0 \
  --lambda_preserve "$LAMBDA_PRESERVE" \
  --train_fraction "$TRAIN_FRACTION" \
  --eval_max_blocks "$EVAL_MAX_BLOCKS" \
  --seed 0 \
  --data_shuffle_seed 1001 \
  --lr 2e-05 \
  --weight_decay 0.1 \
  --micro_batch 4 \
  --grad_accum 4 \
  --eval_batch 8 \
  --out_dir "$OUT_DIR" \
  --save_checkpoints \
  2>&1 | tee "random_layer_freeze_pilot_lam${LAMBDA_PRESERVE}.log"
