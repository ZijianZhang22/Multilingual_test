#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

: "${ANCHOR_CHECKPOINT:?Set ANCHOR_CHECKPOINT to the anchor model directory}"
: "${ADAPTED_CHECKPOINT:?Set ADAPTED_CHECKPOINT to the adapted model directory}"
: "${LAYER:?Set LAYER to the 1-based hidden-state/layer index used by the subspace fit}"

MODEL_TAG="${MODEL_TAG:-custom}"
OUT_ROOT="${OUT_ROOT:-mechanism_runs/semantic_causal_validation/${MODEL_TAG}}"
LANGUAGES="${LANGUAGES:-en zh fr de es}"
RANK="${RANK:-64}"
EXTRACT_BATCH="${EXTRACT_BATCH:-4}"
EVAL_BATCH="${EVAL_BATCH:-4}"
EVAL_MAX_BLOCKS="${EVAL_MAX_BLOCKS:-128}"
RANDOM_DRAWS="${RANDOM_DRAWS:-4}"
MEAN_CORE="${MEAN_CORE:-}"

LAST_DIR="${OUT_ROOT}/last_pool_subspaces"
STEP7_DIR="${OUT_ROOT}/last_pool_drift_isr_partition"
SEM_DIR="${OUT_ROOT}/semantic_specificity"
SWEEP_DIR="${OUT_ROOT}/feature_intervention_sweep"
POOL_DIR="${OUT_ROOT}/pooling_comparison"
LOG_DIR="${OUT_ROOT}/logs"
PACKAGE="${OUT_ROOT}/semantic_causal_validation_results.tar.gz"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Semantic causal validation suite"
echo "anchor      = ${ANCHOR_CHECKPOINT}"
echo "adapted     = ${ADAPTED_CHECKPOINT}"
echo "layer       = ${LAYER}"
echo "languages   = ${LANGUAGES}"
echo "out_root    = ${OUT_ROOT}"
echo "============================================================"

echo
echo "===== [1/4] Fit feature-matched LAST-token subspaces ====="
if [[ -f "${LAST_DIR}/core_subspaces.pt" ]]; then
  echo "LAST-token core subspaces already exist; skipping."
else
  # shellcheck disable=SC2086
  python experiments/retention_subspace_replication/build_core_subspaces.py     --anchor_checkpoint "${ANCHOR_CHECKPOINT}"     --adapted_checkpoint "${ADAPTED_CHECKPOINT}"     --languages ${LANGUAGES}     --layer "${LAYER}"     --rank "${RANK}"     --extract_batch "${EXTRACT_BATCH}"     --pool last     --out_dir "${LAST_DIR}"     2>&1 | tee "${LOG_DIR}/01_build_last_pool.log"
fi

echo
echo "===== [2/4] Re-run Drift/ISR partition rescue with LAST-token bases ====="
if [[ -f "${STEP7_DIR}/partition_rescue_summary.csv" ]]; then
  echo "LAST-token Step7 already exists; skipping."
else
  python experiments/retention_subspace_mechanism/run_step7_drift_isr_partition_rescue.py     --anchor_checkpoint "${ANCHOR_CHECKPOINT}"     --adapted_checkpoint "${ADAPTED_CHECKPOINT}"     --subspace_file "${LAST_DIR}/core_subspaces.pt"     --data_dir invariance_data/wiki     --old_language en     --new_language zh     --ks 16 32     --alphas 0.25 0.5 1.0     --n_random 8     --eval_max_blocks "${EVAL_MAX_BLOCKS}"     --eval_batch "${EVAL_BATCH}"     --out_dir "${STEP7_DIR}"     2>&1 | tee "${LOG_DIR}/02_last_pool_step7.log"
fi

echo
echo "===== [3/4] Semantic specificity / cross-lingual retrieval ====="
python experiments/semantic_causal_validation/analyze_semantic_specificity.py   --probe_features "${LAST_DIR}/features/adapted_probe.pt"   --aligned_features "${LAST_DIR}/features/adapted_aligned.pt"   --probe_data "${LAST_DIR}/data/xnli_probe.jsonl"   --aligned_data "${LAST_DIR}/data/xnli_aligned.jsonl"   --core_subspace_file "${LAST_DIR}/core_subspaces.pt"   --partition_file "${STEP7_DIR}/drift_isr_partition.pt"   --layer "${LAYER}"   --random_draws "${RANDOM_DRAWS}"   --out_dir "${SEM_DIR}"   2>&1 | tee "${LOG_DIR}/03_semantic_specificity.log"

echo
echo "===== [4/4] Equal-energy intervention-strength sweep ====="
python experiments/semantic_causal_validation/run_intervention_strength_sweep.py   --features_file "${LAST_DIR}/features/adapted_probe.pt"   --core_subspace_file "${LAST_DIR}/core_subspaces.pt"   --partition_file "${STEP7_DIR}/drift_isr_partition.pt"   --layer "${LAYER}"   --betas 0.10 0.25 0.50 1.00   --random_draws "${RANDOM_DRAWS}"   --out_dir "${SWEEP_DIR}"   2>&1 | tee "${LOG_DIR}/04_intervention_sweep.log"

if [[ -n "${MEAN_CORE}" && -f "${MEAN_CORE}" ]]; then
  echo
  echo "===== Optional: compare legacy MEAN-pool vs LAST-token bases ====="
  python experiments/semantic_causal_validation/compare_pooling_subspaces.py     --mean_core "${MEAN_CORE}"     --last_core "${LAST_DIR}/core_subspaces.pt"     --out_dir "${POOL_DIR}"     2>&1 | tee "${LOG_DIR}/05_pooling_comparison.log"
fi

echo
echo "===== Package compact results ====="
rm -f "${PACKAGE}"
items=(
  "${LAST_DIR}/summary.json"
  "${STEP7_DIR}/geometry.json"
  "${STEP7_DIR}/principal_alignment_spectrum.csv"
  "${STEP7_DIR}/partition_rescue_summary.csv"
  "${SEM_DIR}/semantic_specificity_summary.csv"
  "${SEM_DIR}/retrieval_by_language_pair.csv"
  "${SWEEP_DIR}/intervention_strength_sweep.csv"
  "${SWEEP_DIR}/rank32_primary_equal_energy.csv"
  "${LOG_DIR}"
)
if [[ -f "${POOL_DIR}/pooling_subspace_overlap.csv" ]]; then
  items+=("${POOL_DIR}/pooling_subspace_overlap.csv")
fi

tar -czf "${PACKAGE}" "${items[@]}"

echo
echo "============================================================"
echo "DONE"
echo "Results package: ${PACKAGE}"
echo "============================================================"
ls -lh "${PACKAGE}"
