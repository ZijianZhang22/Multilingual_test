#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LAMBDA_PRESERVE="${LAMBDA_PRESERVE:-3.0}"\nPILOT_DIR="${PILOT_DIR:-random_layer_freeze_runs/en_to_zh_seed0_lam${LAMBDA_PRESERVE}}"

python experiments/random_layer_freeze_pilot/analyze_mechanisms.py \
  --pilot_dir "$PILOT_DIR" \
  --anchor_checkpoint invariance_runs/sequence_seed0/en__zh/stage1_en \
  --data_dir invariance_data/wiki \
  --old_language en \
  --new_language zh \
  --layers 12 20 24 \
  --probe_file invariance_data/xnli_freeze_pilot.jsonl \
  --probe_per_lang 100 \
  --gradient_max_batches 4 \
  --gradient_max_blocks 32 \
  2>&1 | tee random_layer_freeze_mechanisms.log
