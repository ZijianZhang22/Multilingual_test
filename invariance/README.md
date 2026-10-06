# Invariance PoC: Cross-Lingual Transfer and Forgetting

This directory is intentionally separate from the original pilot. The first goal is diagnostic rather than proposing a new method immediately:

1. Train identical pretrained models under different language orders with matched per-language optimization.
2. Identify layers/subspaces where one shared task predictor works across languages.
3. Freeze that predictor and test whether continual language adaptation makes the invariant representation less accessible.
4. Later correlate invariant drift with forgetting and transfer.

The key design choice is to use XNLI as the invariance task. We do **not** directly apply IRM to next-token prediction, because the token target is inherently language-specific. XNLI gives the same label semantics across language environments.

## 1. Prepare multilingual Wikipedia train/validation splits

```bash
python invariance/prepare_wikipedia_multilang.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --languages en zh fr \
  --train_tokens 1000000 \
  --val_tokens 100000 \
  --block_size 512 \
  --out_dir invariance_data/wiki
```

Each language gets separate `*_train.pt` and `*_val.pt` files. Validation is collected first, and the boundary token buffer is discarded before training data is collected from later articles.

## 2. Train order-controlled checkpoints

```bash
python invariance/train_sequence.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --data_dir invariance_data/wiki \
  --sequences en,zh zh,en \
  --out_dir invariance_runs/sequence_seed0 \
  --seed 0 \
  --micro_batch 4 \
  --grad_accum 4
```

Important control: every language stage uses a **constant LR and a fresh optimizer**. EN and ZH therefore receive the same LR exposure whether they appear first or second. This removes the old across-stage cosine-schedule confound.

Example checkpoints:

```text
invariance_runs/sequence_seed0/en__zh/stage0_base
invariance_runs/sequence_seed0/en__zh/stage1_en
invariance_runs/sequence_seed0/en__zh/stage2_zh
invariance_runs/sequence_seed0/zh__en/stage1_zh
invariance_runs/sequence_seed0/zh__en/stage2_en
```

## 3. Prepare XNLI probe data

```bash
python invariance/prepare_xnli.py \
  --languages en zh fr \
  --train_per_lang 1200 \
  --test_per_lang 1200 \
  --out_file invariance_data/xnli_probe.jsonl
```

XNLI validation examples are used only to fit probes. XNLI test examples are held out for probe evaluation.

## 4. Extract hidden states from the base model

```bash
python invariance/extract_hidden.py \
  --checkpoint invariance_runs/sequence_seed0/en__zh/stage0_base \
  --data_file invariance_data/xnli_probe.jsonl \
  --out_file invariance_features/base.pt \
  --layers 0 4 8 12 16 20 24
```

Hidden-state index 0 is the embedding output. Later indices are transformer-layer outputs. If a requested hidden-state index does not exist, the script fails explicitly instead of silently using the wrong layer.

## 5. Fit IRM-style invariant probes layer by layer

```bash
python invariance/fit_invariant_probe.py \
  --features_file invariance_features/base.pt \
  --out_dir invariance_probes/base \
  --proj_dim 64 \
  --irm_lambda 1.0
```

The probe learns a projection `P` plus one shared XNLI classifier. Each language is treated as an environment.

The script reports:

- shared XNLI task accuracy;
- variance of task risk across languages;
- how well language ID is linearly decoded from the projected representation;
- an exploratory `invariance_score` used only to select a candidate layer.

A strong candidate subspace should have high task accuracy, low cross-language risk variance, and low language decodability.

## 6. Test whether the invariant representation survives continual adaptation

Extract the same hidden-state indices from later checkpoints:

```bash
python invariance/extract_hidden.py \
  --checkpoint invariance_runs/sequence_seed0/en__zh/stage1_en \
  --data_file invariance_data/xnli_probe.jsonl \
  --out_file invariance_features/en__zh_stage1.pt \
  --layers 0 4 8 12 16 20 24

python invariance/extract_hidden.py \
  --checkpoint invariance_runs/sequence_seed0/en__zh/stage2_zh \
  --data_file invariance_data/xnli_probe.jsonl \
  --out_file invariance_features/en__zh_stage2.pt \
  --layers 0 4 8 12 16 20 24
```

Then evaluate the **same frozen base-model probe** on all checkpoints:

```bash
python invariance/evaluate_frozen_probe.py \
  --probe_file invariance_probes/base/probe_layer_12.pt \
  --features_files \
    invariance_features/base.pt \
    invariance_features/en__zh_stage1.pt \
    invariance_features/en__zh_stage2.pt \
  --out_file invariance_analysis/frozen_probe.csv
```

Use the layer selected in `invariance_probes/base/best_probe.json`, not necessarily layer 12.

## Interpretation

If the frozen base-model probe degrades after adaptation but a newly refit probe on the later checkpoint recovers task accuracy, the cross-lingual information may still exist but be reorganized or less accessible in the original invariant coordinates.

If even a newly refit probe cannot recover the task performance, that is stronger evidence of representational information loss.

The intended next analysis is:

```text
language order
    -> invariant-subspace drift
    -> old-language forgetting
    -> future cross-lingual transfer
```

## What this PoC does not claim yet

- The exploratory invariance score is not a theoretical estimator of invariance.
- High probe accuracy does not prove that the causal LM itself uses the projected information.
- Frozen-probe drift is not automatically the same thing as catastrophic forgetting.
- XNLI is only the first task; a publishable study should eventually test more than one shared cross-lingual capability.

Only if this PoC produces a clear signal should we add CKA, gradient alignment, causal interventions, or invariant-preservation training.
