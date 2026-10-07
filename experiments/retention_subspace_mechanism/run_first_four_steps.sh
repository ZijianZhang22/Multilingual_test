#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"
DATA_DIR="${DATA_DIR:-invariance_data/wiki}"
ANCHOR_ROOT="${ANCHOR_ROOT:-invariance_runs/sequence_seed0}"
ANCHOR_REAL="${ANCHOR_REAL:-${ANCHOR_ROOT}/en/stage1_en}"
ANCHOR="${ANCHOR:-${ANCHOR_ROOT}/en__zh/stage1_en}"

STEP1_DIR="${STEP1_DIR:-mechanism_runs/step1_layer_lambda_sweep}"
STEP2_DIR="${STEP2_DIR:-mechanism_runs/step2_layer20_subspaces}"
SUBSPACE_LANGUAGES="${SUBSPACE_LANGUAGES:-en zh fr de es}"
STEP3_DIR="${STEP3_DIR:-mechanism_runs/step3_causal_removal}"
STEP4_DIR="${STEP4_DIR:-mechanism_runs/step4_causal_rescue}"
LOG_DIR="${LOG_DIR:-mechanism_runs/logs_first_four_steps}"

TRAIN_FRACTION="${TRAIN_FRACTION:-0.20}"
LAMBDAS="${LAMBDAS:-5 20 50}"
EVAL_MAX_BLOCKS="${EVAL_MAX_BLOCKS:-128}"
PACKAGE_NAME="${PACKAGE_NAME:-retention_subspace_first_four_steps.tar.gz}"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Retention-Subspace Mechanism Pipeline: Steps 1-4"
echo "model            = $MODEL_NAME"
echo "train_fraction   = $TRAIN_FRACTION"
echo "lambdas          = $LAMBDAS"
echo "eval_max_blocks  = $EVAL_MAX_BLOCKS"
echo "subspace langs   = $SUBSPACE_LANGUAGES"
echo "============================================================"

# ----------------------------------------------------------------------
# Bootstrap EN/ZH Wikipedia data if absent.
# ----------------------------------------------------------------------
if [[ ! -f "$DATA_DIR/en_train.pt" || ! -f "$DATA_DIR/en_val.pt" ||       ! -f "$DATA_DIR/zh_train.pt" || ! -f "$DATA_DIR/zh_val.pt" ]]; then
  echo
  echo "===== Bootstrap: preparing EN/ZH Wikipedia blocks ====="
  python invariance/prepare_wikipedia_multilang.py     --model_name "$MODEL_NAME"     --languages en zh     --train_tokens 1000000     --val_tokens 100000     --block_size 512     --seed 1234     --out_dir "$DATA_DIR"     2>&1 | tee "$LOG_DIR/00_prepare_data.log"
else
  echo "Wikipedia data already exists; skipping."
fi

# ----------------------------------------------------------------------
# Bootstrap stage1 EN anchor if absent.
# ----------------------------------------------------------------------
if [[ ! -f "$ANCHOR_REAL/config.json" && ! -f "$ANCHOR/config.json" ]]; then
  echo
  echo "===== Bootstrap: training EN stage1 anchor ====="
  python invariance/train_sequence.py     --model_name "$MODEL_NAME"     --data_dir "$DATA_DIR"     --sequences en     --eval_languages en zh     --out_dir "$ANCHOR_ROOT"     --seed 0     --lr 2e-5     --weight_decay 0.1     --micro_batch 4     --grad_accum 4     --eval_batch 8     2>&1 | tee "$LOG_DIR/00_train_anchor.log"
fi

if [[ ! -f "$ANCHOR/config.json" ]]; then
  if [[ ! -f "$ANCHOR_REAL/config.json" ]]; then
    echo "ERROR: could not find/create anchor checkpoint." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$ANCHOR")"
  if [[ ! -e "$ANCHOR" && ! -L "$ANCHOR" ]]; then
    REL_TARGET="$(python - <<PY
import os
print(os.path.relpath("$ANCHOR_REAL", os.path.dirname("$ANCHOR")))
PY
)"
    ln -s "$REL_TARGET" "$ANCHOR"
  fi
fi

echo "Anchor: $ANCHOR"
test -f "$ANCHOR/config.json"

# ----------------------------------------------------------------------
# Step 1: normalized layer/lambda sweep, freeze0 only.
# ----------------------------------------------------------------------
echo
echo "===== [1/4] Layer x lambda Pareto sweep ====="
if [[ -f "$STEP1_DIR/results.csv" && -f "$STEP1_DIR/checkpoints/full_ft/config.json" ]]; then
  echo "Step 1 already complete; skipping."
else
  # shellcheck disable=SC2086
  python experiments/retention_subspace_mechanism/run_step1_lambda_sweep.py     --anchor_checkpoint "$ANCHOR"     --data_dir "$DATA_DIR"     --layers 12 20 24     --lambdas $LAMBDAS     --train_fraction "$TRAIN_FRACTION"     --eval_max_blocks "$EVAL_MAX_BLOCKS"     --out_dir "$STEP1_DIR"     2>&1 | tee "$LOG_DIR/01_lambda_sweep.log"
fi

ADAPTED="$STEP1_DIR/checkpoints/full_ft"
test -f "$ADAPTED/config.json"

# ----------------------------------------------------------------------
# Step 2: build Layer-20 candidate mechanism subspaces.
# ----------------------------------------------------------------------
echo
echo "===== [2/4] Build Layer-20 subspaces ====="
if [[ -f "$STEP2_DIR/layer20_subspaces.pt" ]]; then
  echo "Step 2 already complete; skipping."
else
  python experiments/retention_subspace_mechanism/build_step2_layer20_subspaces.py     --anchor_checkpoint "$ANCHOR"     --adapted_checkpoint "$ADAPTED"     --languages $SUBSPACE_LANGUAGES     --layer 20     --rank 64     --inlp_iters 32     --isr_cov_class 0     --vicreg_epochs 300     --out_dir "$STEP2_DIR"     2>&1 | tee "$LOG_DIR/02_build_subspaces.log"
fi

SUBSPACE_FILE="$STEP2_DIR/layer20_subspaces.pt"
test -f "$SUBSPACE_FILE"

# ----------------------------------------------------------------------
# Step 3: causal removal with matched-rank random controls.
# ----------------------------------------------------------------------
echo
echo "===== [3/4] Matched-rank causal removal ====="
if [[ -f "$STEP3_DIR/matched_random_comparison.csv" ]]; then
  echo "Step 3 already complete; skipping."
else
  python experiments/retention_subspace_mechanism/run_step3_causal_removal.py     --anchor_checkpoint "$ANCHOR"     --adapted_checkpoint "$ADAPTED"     --subspace_file "$SUBSPACE_FILE"     --data_dir "$DATA_DIR"     --languages en zh     --strengths 0.25 0.5 1.0     --eval_max_blocks "$EVAL_MAX_BLOCKS"     --out_dir "$STEP3_DIR"     2>&1 | tee "$LOG_DIR/03_causal_removal.log"
fi

# ----------------------------------------------------------------------
# Step 4: causal rescue along the same real/random subspaces.
# ----------------------------------------------------------------------
echo
echo "===== [4/4] Anchor-direction causal rescue ====="
if [[ -f "$STEP4_DIR/matched_random_rescue_comparison.csv" ]]; then
  echo "Step 4 already complete; skipping."
else
  python experiments/retention_subspace_mechanism/run_step4_causal_rescue.py     --anchor_checkpoint "$ANCHOR"     --adapted_checkpoint "$ADAPTED"     --subspace_file "$SUBSPACE_FILE"     --data_dir "$DATA_DIR"     --old_language en     --new_language zh     --alphas 0.25 0.5 1.0     --eval_max_blocks "$EVAL_MAX_BLOCKS"     --out_dir "$STEP4_DIR"     2>&1 | tee "$LOG_DIR/04_causal_rescue.log"
fi

# ----------------------------------------------------------------------
# Compact cross-step summary.
# ----------------------------------------------------------------------
python - <<PY
import csv, json
from pathlib import Path

s1 = Path("$STEP1_DIR")
s2 = Path("$STEP2_DIR")
s3 = Path("$STEP3_DIR")
s4 = Path("$STEP4_DIR")

summary = {}

rows = list(csv.DictReader((s1 / "results.csv").open()))
summary["step1_pareto"] = [
    {
        "condition": r["condition"],
        "forgetting": float(r["forgetting"]),
        "new_language_gain": float(r["new_language_gain"]),
        "pareto_nondominated": str(r["pareto_nondominated"]).lower() == "true",
    }
    for r in rows
]

summary["step2"] = json.loads((s2 / "summary.json").read_text())

remove = list(csv.DictReader((s3 / "matched_random_comparison.csv").open()))
summary["step3_beta1"] = [
    {
        "model_state": r["model_state"],
        "subspace": r["real_subspace"],
        "language": r["language"],
        "causal_excess_loss_vs_random": float(r["causal_excess_loss_vs_random"]),
    }
    for r in remove if abs(float(r["strength"]) - 1.0) < 1e-12
]

rescue = list(csv.DictReader((s4 / "matched_random_rescue_comparison.csv").open()))
summary["step4_alpha1"] = [
    {
        "subspace": r["real_subspace"],
        "old_rescue_excess_vs_random": float(r["old_rescue_excess_vs_random"]),
        "new_language_cost": float(r["real_new_language_cost"]),
        "old_recovery_fraction": float(r["real_old_recovery_fraction"]),
    }
    for r in rescue if abs(float(r["alpha"]) - 1.0) < 1e-12
]

Path("mechanism_runs/first_four_steps_summary.json").write_text(
    json.dumps(summary, indent=2)
)
print(json.dumps(summary, indent=2))
PY

# ----------------------------------------------------------------------
# Package compact results; intentionally omit checkpoints and feature tensors.
# ----------------------------------------------------------------------
rm -f "$PACKAGE_NAME"
tar -czf "$PACKAGE_NAME"   "$STEP1_DIR/results.csv"   "$STEP1_DIR/manifest.json"   "$STEP2_DIR/summary.json"   "$STEP2_DIR/layer20_subspaces.pt"   "$STEP3_DIR/removal_results.csv"   "$STEP3_DIR/matched_random_comparison.csv"   "$STEP3_DIR/manifest.json"   "$STEP4_DIR/rescue_results.csv"   "$STEP4_DIR/matched_random_rescue_comparison.csv"   "$STEP4_DIR/manifest.json"   mechanism_runs/first_four_steps_summary.json   "$LOG_DIR"

echo
echo "============================================================"
echo "ALL FOUR STEPS COMPLETE"
echo "Step 1: $STEP1_DIR/results.csv"
echo "Step 2: $STEP2_DIR/summary.json"
echo "Step 3: $STEP3_DIR/matched_random_comparison.csv"
echo "Step 4: $STEP4_DIR/matched_random_rescue_comparison.csv"
echo "Upload: $PACKAGE_NAME"
echo "============================================================"
ls -lh "$PACKAGE_NAME"
