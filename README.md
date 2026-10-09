## Route B: controlled continual pretraining

Run `bash run_route_b_forgetting.sh` on a RunPod PyTorch GPU image to train
SmolLM2-360M independently on Chinese-only, English-only and 50/50 mixed
Wikipedia, evaluate fixed held-out text, and generate checkpoint heatmaps.
See [the Route B guide](experiments/route_b_forgetting/README.md) for budgets,
output files and interpretation limits.

# Multilingual Training History -> Future Language Plasticity (Pilot)

## Fixed-input activation heatmaps

Run `bash run_activation_heatmaps.sh --pair all --pool all` to compare the two public
TinyLlama v1.1 English-dominant / Chinese-English branches on the same 100 inputs.
Use `--pair all` for all three small-model pairs, or `--before` / `--after` for
your own checkpoints. The script generates PNGs, CSVs, an HTML gallery, and a
downloadable result ZIP. See [the heatmap guide](experiments/activation_heatmaps/README.md)
for RunPod instructions and interpretation limits.

Hypothesis: holding source languages, source examples, token budget, compute, and target data fixed, does the **schedule** of multilingual exposure change how quickly the model adapts to the same later target language?

Default experiment:

- Base model: `Qwen/Qwen2.5-0.5B`
- A = English Wikipedia
- B = Chinese Wikipedia
- C = Swahili Wikipedia
- 5M model tokens per source language
- 2M target training tokens + 250K target validation tokens
- context length 512
- branches: `base`, `mix`, `en_zh`, `zh_en`

## Install

```bash
pip install -r requirements.txt
```

## Prepare data

```bash
python prepare_wikipedia.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --source_tokens 5000000 \
  --target_train_tokens 2000000 \
  --target_val_tokens 250000 \
  --block_size 512 \
  --out_dir prepared
```

## One-seed sanity run

```bash
python run_pilot.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --data_dir prepared \
  --out_dir runs/seed0 \
  --seed 0 \
  --micro_batch 4 \
  --grad_accum 4 \
  --gradient_checkpointing
```

If memory is plentiful, omit `--gradient_checkpointing`.

## Then 3 seeds if the curves separate

```bash
for s in 0 1 2; do
  python run_pilot.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --data_dir prepared \
    --out_dir runs/seed${s} \
    --seed ${s} \
    --micro_batch 4 \
    --grad_accum 4 \
    --gradient_checkpointing
done
```

## Analyze

```bash
python analyze_results.py \
  --metrics_glob "runs/seed*/metrics.csv" \
  --out_dir analysis
```

Optional threshold metric:

```bash
python analyze_results.py \
  --metrics_glob "runs/seed*/metrics.csv" \
  --out_dir analysis \
  --threshold 2.5
```

Outputs include `learning_curves.png`, per-seed summaries, and aggregate summaries.

## Important controls implemented

1. Every branch reloads the identical pretrained checkpoint.
2. `mix`, `en_zh`, and `zh_en` use the exact same EN and ZH blocks.
3. EN and ZH contribute equal model-token budgets.
4. One optimizer/scheduler spans the whole source stage; sequential branches do not reset the optimizer at the EN/ZH boundary.
5. All branches use the same target-language block order within a seed.
6. Target adaptation always starts with a **fresh optimizer and LR scheduler**, preventing source optimizer state from becoming the explanation.
7. All branches use the same fixed held-out target validation set.
8. The key outcome is the whole target learning curve, not only final loss.

`normalized_area_under_loss_curve` is lower-is-better because the y-axis is loss.

## Interpretation

Interesting pilot outcomes include:

- `mix` learns C faster than both sequential branches;
- the average of `en_zh` and `zh_en` differs from `mix`;
- `en_zh` differs from `zh_en`, suggesting an order effect;
- final C performance converges but early learning speed remains different.

Qwen2.5 is already multilingual, so the claim is about **future adaptation/plasticity**, not learning Swahili literally from zero.

## Strong next control

If `en_zh` and `zh_en` differ, recency is the obvious alternative explanation. Next test histories that end in the same language, e.g. `A1 -> B -> A2 -> C` versus `B -> A1 -> A2 -> C`, while matching examples and total source tokens.
