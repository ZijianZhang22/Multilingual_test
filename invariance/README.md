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
  --eval_languages en zh fr \
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


## 7. Validate that the representation is really cross-lingual

The basic layer summary is not enough. A useful invariant representation should survive stronger tests.

### 7.1 Leave one language out

Fit the representation using only two languages and evaluate on the third language that the probe never saw:

```bash
python invariance/validate_leave_one_language_out.py \
  --features_file invariance_features/base.pt \
  --out_file invariance_analysis/leave_one_language_out.csv \
  --proj_dim 64 \
  --irm_lambdas 0 1
```

This runs both ERM (`irm_lambda=0`) and IRM (`irm_lambda=1`) under the same protocol. Examples:

```text
train EN + ZH -> test FR
train EN + FR -> test ZH
train ZH + FR -> test EN
```

A candidate invariant representation is more convincing if held-out-language task accuracy remains strong, especially if IRM improves over the ERM control.

### 7.2 Frozen probe versus re-fitted probe

After extracting features from later checkpoints, compare the original frozen base probe to a freshly re-fitted probe at exactly the same layer:

```bash
python invariance/validate_frozen_vs_refit.py \
  --reference_probe invariance_probes/base/probe_layer_12.pt \
  --features_files \
    invariance_features/base.pt \
    invariance_features/en__zh_stage1.pt \
    invariance_features/en__zh_stage2.pt \
    invariance_features/zh__en_stage1.pt \
    invariance_features/zh__en_stage2.pt \
  --out_file invariance_analysis/frozen_vs_refit.csv
```

Again, use the actual layer selected in `best_probe.json`, not necessarily layer 12.

Interpretation:

- frozen probe drops, re-fitted probe recovers: the information may still exist but the old invariant coordinates became less accessible;
- both frozen and re-fitted probes drop: stronger evidence that shared task information itself degraded;
- neither drops: this particular invariant representation is stable under the language adaptation.

The script also writes:

```text
invariance_analysis/frozen_vs_refit_summary.csv
```

with mean frozen accuracy, mean re-fitted accuracy, and the recovery gap for each checkpoint.

### 7.3 Link invariant drift to language forgetting

The sequence trainer now records stage-0 losses as well as later-stage losses. It can also evaluate a language that is not in the training sequence, for example FR:

```bash
python invariance/train_sequence.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --data_dir invariance_data/wiki \
  --sequences en,zh zh,en \
  --eval_languages en zh fr \
  --out_dir invariance_runs/sequence_seed0 \
  --seed 0
```

Once `frozen_vs_refit_summary.csv` exists:

```bash
python invariance/analyze_invariance_forgetting.py \
  --sequence_metrics invariance_runs/sequence_seed0/all_sequence_metrics.csv \
  --invariance_summary invariance_analysis/frozen_vs_refit_summary.csv \
  --out_dir invariance_analysis/forgetting_link
```

This computes, for each sequence/stage/language:

```text
forgetting_loss_delta
invariant_access_drop
refit_information_drop
recovery_gap_accuracy
```

and reports their Pearson correlations.

With only EN/ZH orders and one seed, these correlations are **diagnostic only**. A paper-level claim needs more language transitions and multiple seeds.

## Validation checklist

A candidate invariant representation should ideally pass all of these:

1. high shared XNLI task accuracy;
2. low risk variance across languages;
3. reduced language-ID decodability;
4. non-trivial leave-one-language-out transfer;
5. frozen-probe drift that systematically relates to forgetting;
6. a useful distinction between frozen-probe loss and re-fitted-probe recovery.

Only after these diagnostics show a repeatable signal should we add invariant-preserving continual training as a causal intervention.


## 8. Compare several representation-discovery / invariance methods

A unified benchmark is available in:

```text
invariance/benchmark_representation_methods.py
```

It currently compares five methods under the same hidden-state features and evaluation protocol:

- `ERM`: shared low-dimensional task projection without an invariance penalty;
- `IRM`: shared task projection with the IRM gradient penalty;
- `V-REx`: minimizes mean task risk plus variance of risk across language environments;
- `DANN`: adversarially preserves XNLI information while making language ID harder to decode;
- `INLP`: iteratively removes linearly decodable language directions, then fits the task head on the erased representation.

The benchmark reports:

```text
mean_task_accuracy
language_probe_accuracy
risk_variance
leave-one-language-out accuracy
```

and saves the learned representation artifacts for later continual-learning analysis.

### Quick experiment: Layer 12 only

Because the first IRM/LOO experiment showed the strongest cross-lingual signal around layer 12, start with:

```bash
python invariance/benchmark_representation_methods.py \
  --features_file invariance_features/base.pt \
  --out_dir invariance_analysis/method_benchmark_layer12 \
  --layers 12 \
  --methods erm irm vrex dann inlp \
  --proj_dim 64 \
  --epochs 250 \
  --irm_lambda 1.0 \
  --vrex_lambda 10.0 \
  --dann_lambda 1.0 \
  --inlp_iters 8
```

Important outputs:

```text
invariance_analysis/method_benchmark_layer12/all_language_summary.csv
invariance_analysis/method_benchmark_layer12/leave_one_out.csv
invariance_analysis/method_benchmark_layer12/leave_one_out_aggregate.csv
```

A useful method should ideally keep XNLI task accuracy high, push fresh language-probe accuracy toward chance (1/3 for EN/ZH/FR), and retain strong leave-one-language-out accuracy.

### Full layer sweep

If the layer-12 comparison is informative, run:

```bash
python invariance/benchmark_representation_methods.py \
  --features_file invariance_features/base.pt \
  --out_dir invariance_analysis/method_benchmark_all \
  --methods erm irm vrex dann inlp \
  --proj_dim 64 \
  --epochs 250
```

This reuses the already extracted `base.pt`; no model forward passes are required.

### Why these methods

The methods intentionally test different notions of invariance:

```text
IRM   : one predictive rule should work across language environments
V-REx : task risk should be stable across language environments
DANN  : language identity should be hard to decode while task information remains
INLP  : explicitly erase linearly decodable language directions
```

Therefore, agreement among several methods around the same layer/subspace is stronger evidence than relying on the IRM score alone.
