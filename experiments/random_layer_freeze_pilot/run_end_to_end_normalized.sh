#!/usr/bin/env bash
set -euo pipefail

# End-to-end normalized layer-preservation experiment.
# Intended usage on a fresh RunPod after cloning this repository and checking out zijian_test.
#
# Pipeline:
#   1) optional dependency install
#   2) prepare EN/ZH Wikipedia blocks
#   3) train EN-only stage1 anchor
#   4) create expected stage1_en symlink
#   5) run normalized preservation controls (L12/L20/L24 x freeze0/freeze100 + full FT)
#   6) run mechanism analysis
#   7) run 32-batch gradient analysis on full_ft and freeze100 models
#   8) package compact results for upload
#
# All steps are resumable: existing completed outputs are skipped.

cd "$(git rev-parse --show-toplevel)"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"
LAMBDA_PRESERVE="${LAMBDA_PRESERVE:-20}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.20}"
EVAL_MAX_BLOCKS="${EVAL_MAX_BLOCKS:-128}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

DATA_DIR="${DATA_DIR:-invariance_data/wiki}"
ANCHOR_BASE_DIR="${ANCHOR_BASE_DIR:-invariance_runs/sequence_seed0}"
ANCHOR_REAL="${ANCHOR_REAL:-${ANCHOR_BASE_DIR}/en/stage1_en}"
ANCHOR_EXPECTED="${ANCHOR_EXPECTED:-${ANCHOR_BASE_DIR}/en__zh/stage1_en}"

OUT_DIR="${OUT_DIR:-random_layer_freeze_runs/en_to_zh_seed0_norm_lam${LAMBDA_PRESERVE}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
GRAD32_DIR="${GRAD32_DIR:-${OUT_DIR}/mechanism/gradient32}"
PACKAGE_NAME="${PACKAGE_NAME:-normalized_layer_preservation_seed0_lam${LAMBDA_PRESERVE}_results.tar.gz}"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Normalized Layer Preservation: End-to-End"
echo "model              = $MODEL_NAME"
echo "lambda             = $LAMBDA_PRESERVE"
echo "train_fraction     = $TRAIN_FRACTION"
echo "eval_max_blocks    = $EVAL_MAX_BLOCKS"
echo "data_dir           = $DATA_DIR"
echo "anchor             = $ANCHOR_EXPECTED"
echo "out_dir            = $OUT_DIR"
echo "============================================================"

# ----------------------------------------------------------------------
# 0. Environment
# ----------------------------------------------------------------------
if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo
  echo "===== [0/7] Installing dependencies ====="
  python -m pip install -r requirements.txt 2>&1 | tee "$LOG_DIR/00_install.log"
else
  echo
  echo "===== [0/7] Dependency install skipped ====="
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
else:
    raise SystemExit("CUDA GPU is required.")
PY

# ----------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------
echo
echo "===== [1/7] Preparing EN/ZH Wikipedia data ====="
if [[ -f "$DATA_DIR/en_train.pt" && -f "$DATA_DIR/en_val.pt" &&       -f "$DATA_DIR/zh_train.pt" && -f "$DATA_DIR/zh_val.pt" ]]; then
  echo "Data already exists; skipping."
else
  python invariance/prepare_wikipedia_multilang.py     --model_name "$MODEL_NAME"     --languages en zh     --train_tokens 1000000     --val_tokens 100000     --block_size 512     --seed 1234     --out_dir "$DATA_DIR"     2>&1 | tee "$LOG_DIR/01_prepare_data.log"
fi

# ----------------------------------------------------------------------
# 2. EN stage-1 anchor
# ----------------------------------------------------------------------
echo
echo "===== [2/7] Building EN stage1 anchor ====="
if [[ -f "$ANCHOR_REAL/config.json" ]]; then
  echo "Anchor already exists at $ANCHOR_REAL; skipping training."
else
  python invariance/train_sequence.py     --model_name "$MODEL_NAME"     --data_dir "$DATA_DIR"     --sequences en     --eval_languages en zh     --out_dir "$ANCHOR_BASE_DIR"     --seed 0     --lr 2e-5     --weight_decay 0.1     --micro_batch 4     --grad_accum 4     --eval_batch 8     2>&1 | tee "$LOG_DIR/02_train_anchor.log"
fi

if [[ ! -f "$ANCHOR_REAL/config.json" ]]; then
  echo "ERROR: anchor was not created at $ANCHOR_REAL" >&2
  exit 1
fi

# Create the path expected by the existing pilot scripts without duplicating checkpoint files.
mkdir -p "$(dirname "$ANCHOR_EXPECTED")"
if [[ -L "$ANCHOR_EXPECTED" ]]; then
  true
elif [[ -e "$ANCHOR_EXPECTED" ]]; then
  echo "Expected anchor path already exists as a real file/directory; leaving it unchanged."
else
  REL_TARGET="$(python - <<PY
import os
print(os.path.relpath("$ANCHOR_REAL", os.path.dirname("$ANCHOR_EXPECTED")))
PY
)"
  ln -s "$REL_TARGET" "$ANCHOR_EXPECTED"
fi

echo "Anchor ready:"
ls -ld "$ANCHOR_EXPECTED"
test -f "$ANCHOR_EXPECTED/config.json"

# ----------------------------------------------------------------------
# 3. Normalized preservation experiment
# ----------------------------------------------------------------------
echo
echo "===== [3/7] Running normalized freeze0/freeze100 controls ====="
if [[ -f "$OUT_DIR/results.csv" ]]; then
  echo "Main experiment results already exist; skipping."
else
  LAMBDA_PRESERVE="$LAMBDA_PRESERVE"   TRAIN_FRACTION="$TRAIN_FRACTION"   EVAL_MAX_BLOCKS="$EVAL_MAX_BLOCKS"   OUT_DIR="$OUT_DIR"   bash experiments/random_layer_freeze_pilot/run_normalized_control.sh   2>&1 | tee "$LOG_DIR/03_normalized_control.log"
fi

test -f "$OUT_DIR/results.csv"

echo
echo "===== Main result summary ====="
python - <<PY
import csv
from pathlib import Path
p = Path("$OUT_DIR/results.csv")
rows = list(csv.DictReader(p.open()))
for r in sorted(rows, key=lambda x: float(x["forgetting"])):
    print(
        f'{r["condition"]:34s} '
        f'forget={float(r["forgetting"]):+.6f} '
        f'gain={float(r["new_language_gain"]):+.6f} '
        f'pres={float(r["mean_preserve_loss"]):.6f} '
        f'raw={float(r.get("mean_raw_preserve_mse") or 0):.6f} '
        f'E={float(r.get("mean_anchor_energy") or 0):.6f}'
    )
PY

# ----------------------------------------------------------------------
# 4. Mechanism analysis
# ----------------------------------------------------------------------
echo
echo "===== [4/7] Running mechanism analysis ====="
if [[ -f "$OUT_DIR/mechanism/mechanism_summary.csv" ]]; then
  echo "Mechanism summary already exists; skipping."
else
  LAMBDA_PRESERVE="$LAMBDA_PRESERVE"   PILOT_DIR="$OUT_DIR"   bash experiments/random_layer_freeze_pilot/analyze_normalized_control.sh   2>&1 | tee "$LOG_DIR/04_mechanisms.log"
fi

# ----------------------------------------------------------------------
# 5. Gradient32 on the four most informative checkpoints
# ----------------------------------------------------------------------
echo
echo "===== [5/7] Running gradient32 analyses ====="
mkdir -p "$GRAD32_DIR"

LAM_TAG="$(python - <<PY
x=float("$LAMBDA_PRESERVE")
print(("%g" % x).replace(".", "p"))
PY
)"

declare -a CONDITIONS=(
  "full_ft"
  "layer12_freeze100_lam${LAM_TAG}_norm"
  "layer20_freeze100_lam${LAM_TAG}_norm"
  "layer24_freeze100_lam${LAM_TAG}_norm"
)

for COND in "${CONDITIONS[@]}"; do
  CKPT="$OUT_DIR/checkpoints/$COND"
  OUT_CSV="$GRAD32_DIR/$COND.csv"
  if [[ -f "$OUT_CSV" ]]; then
    echo "gradient32 exists for $COND; skipping."
    continue
  fi
  if [[ ! -f "$CKPT/config.json" ]]; then
    echo "WARNING: missing checkpoint $CKPT; skipping gradient32 for $COND" >&2
    continue
  fi
  echo "--- gradient32: $COND ---"
  python invariance/analyze_gradient_interference.py     --checkpoint "$CKPT"     --data_dir "$DATA_DIR"     --old_language en     --new_language zh     --layers 12 20 24     --batch_size 2     --max_batches 32     --max_blocks 128     --seed 0     --out_file "$OUT_CSV"     2>&1 | tee "$LOG_DIR/05_gradient32_${COND}.log"
done

# ----------------------------------------------------------------------
# 6. Compact summary
# ----------------------------------------------------------------------
echo
echo "===== [6/7] Writing compact experiment summary ====="
python - <<PY
import csv, json
from pathlib import Path

out = Path("$OUT_DIR")
rows = list(csv.DictReader((out/"results.csv").open()))
summary = {
    "lambda_preserve": "$LAMBDA_PRESERVE",
    "train_fraction": "$TRAIN_FRACTION",
    "conditions": [],
}
for r in rows:
    summary["conditions"].append({
        "condition": r["condition"],
        "forgetting": float(r["forgetting"]),
        "new_language_gain": float(r["new_language_gain"]),
        "freeze_ratio": float(r["freeze_ratio_requested"]),
        "target_layer": r["target_layer_1based"],
        "mean_preserve_loss": float(r["mean_preserve_loss"]),
        "mean_raw_preserve_mse": float(r.get("mean_raw_preserve_mse") or 0),
        "mean_anchor_energy": float(r.get("mean_anchor_energy") or 0),
    })

grad_dir = out/"mechanism"/"gradient32"
grad = {}
if grad_dir.exists():
    for p in sorted(grad_dir.glob("*.csv")):
        grad[p.stem] = list(csv.DictReader(p.open()))
summary["gradient32_files"] = sorted(grad)

(out/"compact_summary.json").write_text(json.dumps(summary, indent=2))
print(out/"compact_summary.json")
PY

# ----------------------------------------------------------------------
# 7. Package compact outputs (do NOT include model checkpoints/features)
# ----------------------------------------------------------------------
echo
echo "===== [7/7] Packaging results ====="
rm -f "$PACKAGE_NAME"

tar -czf "$PACKAGE_NAME"   "$OUT_DIR/results.csv"   "$OUT_DIR/results_partial.csv"   "$OUT_DIR/manifest.json"   "$OUT_DIR/compact_summary.json"   "$OUT_DIR/mechanism/parameter_drift.csv"   "$OUT_DIR/mechanism/cka_all.csv"   "$OUT_DIR/mechanism/gradient_all.csv"   "$OUT_DIR/mechanism/mechanism_summary.csv"   "$GRAD32_DIR"   "$LOG_DIR"   2>/dev/null || {
    echo "Some optional files were missing; creating fallback package."
    tar -czf "$PACKAGE_NAME"       "$OUT_DIR/results.csv"       "$OUT_DIR/manifest.json"       "$OUT_DIR/compact_summary.json"       "$OUT_DIR/mechanism"       "$LOG_DIR"
  }

echo
echo "============================================================"
echo "ALL DONE"
echo "Main results:      $OUT_DIR/results.csv"
echo "Mechanism summary: $OUT_DIR/mechanism/mechanism_summary.csv"
echo "Gradient32:        $GRAD32_DIR"
echo "Upload package:    $PACKAGE_NAME"
echo "============================================================"
ls -lh "$PACKAGE_NAME"
