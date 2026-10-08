#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-invariance_data/wiki}"
OUT_ROOT="${OUT_ROOT:-replication_runs/qwen_scale_seed}"
SEEDS="${SEEDS:-0 1 2}"
MODELS="${MODELS:-1.5B 3B}"
N_RANDOM="${N_RANDOM:-8}"
EVAL_MAX_BLOCKS="${EVAL_MAX_BLOCKS:-128}"
RANK="${RANK:-64}"

cd "$ROOT_DIR"

run_job () {
  local tag="$1"
  local model="$2"
  local seed="$3"
  local micro="$4"
  local accum="$5"
  local eval_batch="$6"
  local extract_batch="$7"

  echo "============================================================"
  echo "MODEL=$tag  SEED=$seed"
  echo "============================================================"

  python -u experiments/retention_subspace_replication/run_one_replication.py     --model_name "$model"     --model_tag "$tag"     --seed "$seed"     --data_dir "$DATA_DIR"     --out_root "$OUT_ROOT"     --rank "$RANK"     --n_random "$N_RANDOM"     --micro_batch "$micro"     --grad_accum "$accum"     --eval_batch "$eval_batch"     --extract_batch "$extract_batch"     --eval_max_blocks "$EVAL_MAX_BLOCKS"
}

for tag in $MODELS; do
  case "$tag" in
    1.5B)
      model="Qwen/Qwen2.5-1.5B"
      micro=1
      accum=16
      eval_batch=4
      extract_batch=4
      ;;
    3B)
      model="Qwen/Qwen2.5-3B"
      micro=1
      accum=16
      eval_batch=2
      extract_batch=2
      ;;
    *)
      echo "Unknown model tag: $tag (supported: 1.5B 3B)" >&2
      exit 2
      ;;
  esac

  for seed in $SEEDS; do
    run_job "$tag" "$model" "$seed" "$micro" "$accum" "$eval_batch" "$extract_batch"
  done
done

python -u experiments/retention_subspace_replication/summarize_replications.py --root "$OUT_ROOT"
echo "All requested replications finished."
