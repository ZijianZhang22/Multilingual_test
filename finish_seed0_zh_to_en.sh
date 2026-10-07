#!/usr/bin/env bash
set -euo pipefail

# Finish seed0 / ZH -> EN from the current breakpoint.
# Assumes:
#   repo: /Multilingual_test
#   completed already:
#     full_ft
#     lang: 0.3 1 3 5 10 20
#     shared64: 0.3 1 3 5 10
#   and the corresponding lam=10 hidden features already exist.

ROOT=/Multilingual_test
cd "$ROOT"

BASE="invariance_runs/causal_representation_suite/seed0/zh_to_en"
ANALYSIS="invariance_analysis/causal_representation_suite/seed0/zh_to_en"
LANG_SUB="invariance_analysis/causal_representation_suite/reference_inlp_language_subspace.pt"
TRANSFER_SUB="invariance_analysis/causal_representation_suite/transferable_rank64.pt"
ANCHOR="invariance_runs/sequence_seed0/zh__en/stage1_zh"
PROBE="invariance_data/xnli_probe.jsonl"

mkdir -p "$BASE" "$ANALYSIS/manipulation/features" logs

echo "============================================================"
echo "[1/6] Finish missing shared64 lambda=20"
echo "============================================================"

python invariance/train_representation_preservation.py \
  --anchor_checkpoint "$ANCHOR" \
  --subspace_file "$LANG_SUB" \
  --transferable_subspace_file "$TRANSFER_SUB" \
  --data_dir invariance_data/wiki \
  --old_language zh \
  --new_language en \
  --eval_languages en zh \
  --methods shared64 \
  --lambdas 20.0 \
  --layer 12 \
  --shared64_rank 64 \
  --seed 0 \
  --data_shuffle_seed 1000 \
  --lr 2e-05 \
  --weight_decay 0.1 \
  --micro_batch 4 \
  --grad_accum 4 \
  --eval_batch 8 \
  --out_dir "$BASE/resume_shared64_lam20" \
  2>&1 | tee logs/zh_to_en_shared64_lam20.log

echo "============================================================"
echo "[2/6] Finish transfer64/shared/full lambda sweep (18 runs)"
echo "============================================================"

python invariance/train_representation_preservation.py \
  --anchor_checkpoint "$ANCHOR" \
  --subspace_file "$LANG_SUB" \
  --transferable_subspace_file "$TRANSFER_SUB" \
  --data_dir invariance_data/wiki \
  --old_language zh \
  --new_language en \
  --eval_languages en zh \
  --methods transfer64 shared full \
  --lambdas 0.3 1.0 3.0 5.0 10.0 20.0 \
  --layer 12 \
  --shared64_rank 64 \
  --seed 0 \
  --data_shuffle_seed 1000 \
  --lr 2e-05 \
  --weight_decay 0.1 \
  --micro_batch 4 \
  --grad_accum 4 \
  --eval_batch 8 \
  --out_dir "$BASE/resume_remaining" \
  --feature_extract_data "$PROBE" \
  --feature_extract_dir "$ANALYSIS/manipulation/features" \
  --feature_extract_methods transfer64 shared full \
  --feature_extract_lambdas 10.0 \
  --feature_extract_batch 16 \
  2>&1 | tee logs/zh_to_en_remaining_sweep.log

echo "============================================================"
echo "[3/6] Run lambda=3 learning curves for all six methods"
echo "============================================================"

python invariance/train_representation_preservation.py \
  --anchor_checkpoint "$ANCHOR" \
  --subspace_file "$LANG_SUB" \
  --transferable_subspace_file "$TRANSFER_SUB" \
  --data_dir invariance_data/wiki \
  --old_language zh \
  --new_language en \
  --eval_languages en zh \
  --methods full_ft lang shared64 transfer64 shared full \
  --lambdas 3.0 \
  --curve_fractions 0.1 0.25 0.5 0.75 1.0 \
  --layer 12 \
  --shared64_rank 64 \
  --seed 0 \
  --data_shuffle_seed 1000 \
  --lr 2e-05 \
  --weight_decay 0.1 \
  --micro_batch 4 \
  --grad_accum 4 \
  --eval_batch 8 \
  --out_dir "$BASE/curves_lam3" \
  2>&1 | tee logs/zh_to_en_curves_lam3.log

echo "============================================================"
echo "[4/6] Extract anchor hidden features"
echo "============================================================"

python invariance/extract_hidden.py \
  --checkpoint "$ANCHOR" \
  --data_file "$PROBE" \
  --out_file "$ANALYSIS/manipulation/features/anchor.pt" \
  --layers 12 \
  --batch_size 16 \
  2>&1 | tee logs/zh_to_en_extract_anchor.log

echo "============================================================"
echo "[5/6] Analyze intervention drift"
echo "============================================================"

# These first three should already exist from the interrupted original run:
#   full_ft.pt
#   lang_lam10.pt
#   shared64_lam10.pt
# These last three are created in step 2:
#   transfer64_lam10.pt
#   shared_lam10.pt
#   full_lam10.pt

python invariance/analyze_intervention_drift.py \
  --anchor_features "$ANALYSIS/manipulation/features/anchor.pt" \
  --intervention_features \
    "$ANALYSIS/manipulation/features/full_ft.pt" \
    "$ANALYSIS/manipulation/features/lang_lam10.pt" \
    "$ANALYSIS/manipulation/features/shared64_lam10.pt" \
    "$ANALYSIS/manipulation/features/transfer64_lam10.pt" \
    "$ANALYSIS/manipulation/features/shared_lam10.pt" \
    "$ANALYSIS/manipulation/features/full_lam10.pt" \
  --language_subspace_file "$LANG_SUB" \
  --transferable_subspace_file "$TRANSFER_SUB" \
  --out_file "$ANALYSIS/manipulation/intervention_drift.csv" \
  --layer 12 \
  2>&1 | tee logs/zh_to_en_intervention_drift.log

echo "============================================================"
echo "[6/6] Run extended mechanism analysis for seed0, both dirs"
echo "============================================================"

if [[ -f invariance/run_extended_mechanism_analysis.py ]]; then
  python invariance/run_extended_mechanism_analysis.py \
    --seeds 0 \
    --directions en:zh zh:en \
    2>&1 | tee logs/extended_mechanism_analysis_seed0.log
else
  echo "WARNING: invariance/run_extended_mechanism_analysis.py not found."
  echo "Skipping extended mechanism analysis."
fi

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Key outputs:"
echo "  $BASE/resume_shared64_lam20/intervention_metrics.csv"
echo "  $BASE/resume_remaining/intervention_metrics.csv"
echo "  $BASE/curves_lam3/intervention_metrics.csv"
echo "  $BASE/curves_lam3/learning_curves.csv"
echo "  $ANALYSIS/manipulation/intervention_drift.csv"
echo "  logs/extended_mechanism_analysis_seed0.log"
