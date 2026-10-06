#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for the multilingual forgetting experiment.
#
# Usage:
#   bash run_forgetting.sh
#
# Optional overrides:
#   MODEL=Qwen/Qwen2.5-1.5B DATA_DIR=prepared_1p5b_medium \
#   OUT_DIR=runs/1p5b_forgetting_seed0 SESSION=forgetting_1p5b \
#   bash run_forgetting.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
DATA_DIR="${DATA_DIR:-prepared_medium}"
OUT_DIR="${OUT_DIR:-runs/medium_forgetting_seed0}"
SESSION="${SESSION:-forgetting_all}"
SEED="${SEED:-0}"
MICRO_BATCH="${MICRO_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is not installed."
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: data directory '$DATA_DIR' does not exist."
  echo "Prepare the data first, then rerun this script."
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists."
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "Starting experiment in tmux session: $SESSION"
echo "Model:    $MODEL"
echo "Data:     $DATA_DIR"
echo "Output:   $OUT_DIR"
echo "Seed:     $SEED"
echo

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && \
   python run_pilot.py \
     --model_name '$MODEL' \
     --data_dir '$DATA_DIR' \
     --out_dir '$OUT_DIR' \
     --branches base mix en_zh zh_en \
     --seed '$SEED' \
     --micro_batch '$MICRO_BATCH' \
     --grad_accum '$GRAD_ACCUM' \
     --eval_marks 0 25000 50000 100000 250000 500000 \
     --save_models \
     2>&1 | tee '$OUT_DIR/train.log'"

echo "Started successfully."
echo
echo "Attach:      tmux attach -t $SESSION"
echo "Watch log:   tail -f $OUT_DIR/train.log"
echo "Check GPU:   watch -n 2 nvidia-smi"
echo "Check proc:  pgrep -af run_pilot.py"
