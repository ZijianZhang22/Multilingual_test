# Qwen2.5-7B on one A100 80GB

Dedicated full-parameter BF16 multilingual continual-adaptation workflow, based on the existing zijian_test replication. This folder leaves all 0.5B/1.5B/3B files unchanged and reuses the existing five-subspace fitter, Step 6 causal energy-matched controls and Step 7 drift-versus-ISR rescue code.

## Optimizations and scientific caveats

- All model parameters are updated. Model weights remain BF16: **not LoRA, not QLoRA**.
- bitsandbytes AdamW8bit optimizer states (optional PagedAdamW8bit), gradient checkpointing, SDPA attention, micro batch 1 with 16 accumulation steps.
- Constant LR 2e-5, weight decay 0.1. Fresh optimizer and reset RNG at EN-to-ZH stage boundary. Original per-stage shuffles and data-selection logic preserved.
- Batch-1 evaluation/extraction, fixed 128-block validation subset, rank 64, eight random controls.
- New results and checkpoints stored in a dedicated output directory, with per-seed resume and manifest.
- **Not numerically exact replication of original AdamW**: 8-bit optimizer states alter optimization. Report this when comparing scales.
- **Not GPU-benchmarked yet**: Step 6/7 use two BF16 models simultaneously plus hooks and repeated forwards; further modifications may be needed if they OOM.
- Included: EN anchor + ZH adaptation, five fresh subspaces, Step 6, Step 7. Excluded: broad Step 1 lambda/freezing sweeps, Step 5 atlas.

## Install, from repository root

~~~bash
git checkout feature/qwen25-7b-a100-8bit
git pull
python -m pip install -r requirements.txt
python -m pip install -r experiments/qwen25_7b_a100/requirements-7b.txt
nvidia-smi
free -h
ls invariance_data/wiki/{en,zh}_{train,val}.pt
~~~

Requires one A100 80GB, CUDA BF16, internet or cached Hugging Face checkpoints and XNLI data, and approximately 80 GB free disk for model downloads, two BF16 checkpoints and features.

The default dataset budget matches the scale-replication runner: about 1M EN training tokens and **20% of ZH train blocks** (typically 0.2M ZH tokens), not 1M ZH tokens. Use --new_train_fraction 1.0 to adapt on all ZH blocks. Keep token budget and evaluation subsets consistent in cross-scale comparisons.

## Launch

~~~bash
# Check commands without requiring a GPU or model downloads
python experiments/qwen25_7b_a100/run_7b_a100.py --dry_run --through step6

# Train and save EN anchor + ZH adapted checkpoints
CUDA_VISIBLE_DEVICES=0 python -u experiments/qwen25_7b_a100/run_7b_a100.py --seed 0 --through training 2>&1 | tee qwen7b_train.log

# Priority first: refit subspaces and test Step 7
CUDA_VISIBLE_DEVICES=0 python -u experiments/qwen25_7b_a100/run_7b_a100.py --seed 0 --through step7 2>&1 | tee qwen7b_step7.log

# Then complete Step 6 (reusing completed stages)
CUDA_VISIBLE_DEVICES=0 python -u experiments/qwen25_7b_a100/run_7b_a100.py --seed 0 --through step6 2>&1 | tee qwen7b_complete.log

# Three independent training seeds, one after another
for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 python -u experiments/qwen25_7b_a100/run_7b_a100.py --seed "$seed" --through step7 || exit 1
done
~~~

If 8-bit optimizer memory is tight, try --optimizer paged_adamw8bit and **a fresh --out_root**. Avoid mixing optimizer types or fractions under existing checkpoints.

## Small GPU smoke test (not paper-quality results)

~~~bash
CUDA_VISIBLE_DEVICES=0 python -u experiments/qwen25_7b_a100/train_7b_a100.py --out_dir replication_runs/qwen25_7b_a100_smoke/seed0/training --max_optimizer_steps 2 --eval_max_blocks 2
~~~

Do not reuse the smoke-test output folder for full experiments.

## Outputs

~~~text
replication_runs/qwen25_7b_a100/seed0/
  training/
    anchor/
    adapted/
    anchor_summary.json
    training_summary.json
  subspaces/
    core_subspaces.pt
    summary.json
  drift_isr_partition/
    partition_rescue_summary.csv
  energy_controls/
    energy_matched_summary.csv
  run_manifest.json
~~~

Relative layer defaults to round(28 * 20/24) = 23 (28 transformer layers); other model scales must use their own fitted projection bases. Before a long Step 6, check free VRAM during Step 7. Eight random directions are controls, not eight independent training replicates.
