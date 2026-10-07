#!/usr/bin/env bash
set -euo pipefail

# Completely separate advanced diagnostics suite.
# It does NOT modify or overwrite retention_subspace_mechanism outputs.

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${OUT_ROOT:-mechanism_runs/advanced_functional_diagnostics_v1}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

ANCHOR="${ANCHOR:-invariance_runs/sequence_seed0/en__zh/stage1_en}"
ADAPTED="${ADAPTED:-mechanism_runs/step1_layer_lambda_sweep/checkpoints/full_ft}"
STEP2_FILE="${STEP2_FILE:-mechanism_runs/step2_layer20_subspaces_v2/layer20_subspaces.pt}"
FEATURES_FILE="${FEATURES_FILE:-mechanism_runs/step2_layer20_subspaces_v2/features/anchor_probe.pt}"
DATA_DIR="${DATA_DIR:-invariance_data/wiki}"

echo "============================================================"
echo " Advanced Functional Diagnostics v1"
echo "============================================================"
echo "anchor      = $ANCHOR"
echo "adapted     = $ADAPTED"
echo "step2 basis = $STEP2_FILE"
echo "features    = $FEATURES_FILE"
echo "out         = $OUT_ROOT"
echo

if [[ ! -d "$ANCHOR" ]]; then
  echo "ERROR: missing anchor checkpoint: $ANCHOR"
  exit 2
fi
if [[ ! -d "$ADAPTED" ]]; then
  echo "ERROR: missing adapted checkpoint: $ADAPTED"
  echo "Run the main mechanism pipeline through Step 1 first."
  exit 2
fi

# ----------------------------------------------------------------------
# A. Spectral / effective-rank + function/output drift
# ----------------------------------------------------------------------
echo "===== [A/4] spectral + output drift ====="
if [[ -f "$OUT_ROOT/spectral_output/spectral_drift_metrics.csv" ]]; then
  echo "SKIP existing spectral diagnostics."
else
  python experiments/advanced_functional_diagnostics/measure_spectral_output_drift.py \
    --anchor_checkpoint "$ANCHOR" \
    --adapted_checkpoint "$ADAPTED" \
    --data_dir "$DATA_DIR" \
    --languages en zh \
    --layers 4 8 12 16 20 24 \
    --max_blocks 32 \
    --batch_size 2 \
    --token_stride 64 \
    --out_dir "$OUT_ROOT/spectral_output" \
    2>&1 | tee "$LOG_DIR/A_spectral_output.log"
fi

# ----------------------------------------------------------------------
# B. Old/new activation-gradient geometry
# ----------------------------------------------------------------------
echo
echo "===== [B/4] activation-gradient subspaces ====="
if [[ -f "$OUT_ROOT/gradient_subspaces/gradient_subspace_metrics.csv" ]]; then
  echo "SKIP existing gradient subspaces."
else
  python experiments/advanced_functional_diagnostics/measure_gradient_subspaces.py \
    --checkpoint "$ADAPTED" \
    --data_dir "$DATA_DIR" \
    --old_language en \
    --new_language zh \
    --layers 4 8 12 16 20 24 \
    --rank 64 \
    --batch_size 2 \
    --max_batches 16 \
    --token_stride 8 \
    --out_dir "$OUT_ROOT/gradient_subspaces" \
    2>&1 | tee "$LOG_DIR/B_gradient_subspaces.log"
fi

# ----------------------------------------------------------------------
# C. Alternative causal language erasure: mean projection vs LEACE
# ----------------------------------------------------------------------
echo
echo "===== [C/4] mean projection + LEACE controls ====="
if [[ ! -f "$FEATURES_FILE" ]]; then
  echo "WARNING: missing $FEATURES_FILE ; skipping concept erasure controls."
else
  if [[ -f "$OUT_ROOT/concept_erasure/intrinsic_erasure.csv" ]]; then
    echo "SKIP existing concept erasure controls."
  else
    python experiments/advanced_functional_diagnostics/compare_concept_erasure.py \
      --features_file "$FEATURES_FILE" \
      --checkpoint "$ADAPTED" \
      --data_dir "$DATA_DIR" \
      --layer 20 \
      --old_language en \
      --new_language zh \
      --strengths 0.5 1.0 \
      --eval_max_blocks 64 \
      --eval_batch 4 \
      --out_dir "$OUT_ROOT/concept_erasure" \
      2>&1 | tee "$LOG_DIR/C_concept_erasure.log"
  fi
fi

# ----------------------------------------------------------------------
# D. Targeted improvement experiments
# ----------------------------------------------------------------------
echo
echo "===== [D/4] targeted subspace interventions ====="
if [[ ! -f "$STEP2_FILE" ]]; then
  echo "WARNING: missing $STEP2_FILE ; skipping targeted interventions."
else
  if [[ -f "$OUT_ROOT/targeted_interventions/results.csv" ]]; then
    echo "SKIP existing targeted intervention results."
  else
    python experiments/advanced_functional_diagnostics/run_targeted_interventions.py \
      --anchor_checkpoint "$ANCHOR" \
      --basis_file "$STEP2_FILE" \
      --subspaces transfer isr_cov isr_multiclass vicreg drift \
      --layer 20 \
      --data_dir "$DATA_DIR" \
      --old_language en \
      --new_language zh \
      --train_fraction 0.20 \
      --preservation_lambdas 5 20 \
      --shield_alphas 0.5 1.0 \
      --out_dir "$OUT_ROOT/targeted_interventions" \
      2>&1 | tee "$LOG_DIR/D_targeted_interventions.log"
  fi
fi

echo
echo "===== Packing compact results ====="
PACKAGE="${PACKAGE:-advanced_functional_diagnostics_v1.tar.gz}"
tar -czf "$PACKAGE" \
  "$OUT_ROOT/spectral_output/spectral_drift_metrics.csv" \
  "$OUT_ROOT/spectral_output/output_drift.json" \
  "$OUT_ROOT/spectral_output/drift_spectra.pt" \
  "$OUT_ROOT/gradient_subspaces/gradient_subspace_metrics.csv" \
  "$OUT_ROOT/gradient_subspaces/gradient_subspaces.pt" \
  "$OUT_ROOT/concept_erasure" \
  "$OUT_ROOT/targeted_interventions/results.csv" \
  "$OUT_ROOT/logs" \
  2>/dev/null || true

echo
echo "DONE"
echo "Main output: $OUT_ROOT"
echo "Package:     $PACKAGE"
