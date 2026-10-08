# Qwen scale + seed replication

This folder is a clean replication package for the retention-subspace mechanism.

## Main question

Does the seed-0 Qwen2.5-0.5B result replicate across:
1. independent training seeds (0, 1, 2), and
2. larger models from the same family (Qwen2.5-1.5B and Qwen2.5-3B)?

Primary effect: old-language rescue in the Drift subspace relative to globally energy-matched, rank-matched random directions.

Secondary subspaces: ISR-Multiclass, Transfer, ISR-Cov.

## Controlled design

For every model/seed:
- old language: EN
- new language: ZH
- same Wiki block data
- LR = 2e-5, AdamW weight decay = 0.1
- new-language train fraction = 20%
- rank = 64
- rescue strengths alpha = 0.25, 0.5, 1.0
- 128 validation blocks by default
- 8 energy-matched random bases per real subspace
- XNLI fitting data seed fixed at 2026
- model training/data-order seed is the replication seed

The intervention layer is chosen before results by relative depth:
20/24 = 0.8333, matching Layer 20 of the original 24-layer 0.5B setup.
The runner reads model depth and uses round(0.8333 * num_hidden_layers).

This avoids cherry-picking a favorable layer in larger models.

## One-click run

From repository root:

```bash
git checkout zijian_test
git pull
bash experiments/retention_subspace_replication/run_all_qwen_replications.sh
```

Default grid:
- Qwen2.5-1.5B, seeds 0/1/2
- Qwen2.5-3B, seeds 0/1/2

Jobs run sequentially on one GPU.

Examples:

```bash
SEEDS="1 2" MODELS="1.5B" bash experiments/retention_subspace_replication/run_all_qwen_replications.sh
```

```bash
SEEDS="0" MODELS="3B" N_RANDOM=8 bash experiments/retention_subspace_replication/run_all_qwen_replications.sh
```

## Multi-GPU launch

Run one job per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -u experiments/retention_subspace_replication/run_one_replication.py --model_name Qwen/Qwen2.5-1.5B --model_tag 1.5B --seed 0
CUDA_VISIBLE_DEVICES=1 python -u experiments/retention_subspace_replication/run_one_replication.py --model_name Qwen/Qwen2.5-1.5B --model_tag 1.5B --seed 1
```

Do not launch multiple full-FT jobs on the same GPU.

## Outputs

```
replication_runs/qwen_scale_seed/<model>/seed<seed>/
  training/
    anchor/
    adapted/
    training_summary.json
  subspaces/
    core_subspaces.pt
    summary.json
  energy_controls/
    energy_matched_summary.csv
    energy_matched_removal_draws.csv
    energy_matched_rescue_draws.csv
  replication_manifest.json
  replication.log
```

After all requested jobs:

```
replication_runs/qwen_scale_seed/replication_summary.csv
```

## Primary success criterion

For each independently trained model seed, especially alpha=0.25 and 0.50:
- Drift old-language rescue > energy-matched random mean.
- Same sign across seeds.
- New-language loss cost stays small relative to acquired ZH gain.

Do not use random-draw SD as a formal p-value. The independently trained model seed is the replication unit.

## GPU note

This is full fine-tuning with standard AdamW. Qwen2.5-3B can need a high-memory GPU because optimizer states dominate memory. A100 80GB-class hardware is the safest target. If 3B OOMs, do not silently switch only that model to LoRA or quantization; that changes the protocol. Move the 3B job to a larger-memory GPU or define a second protocol consistently.
