# Random Layer Freezing Pilot

## Goal

Test whether a potentially retention-critical layer can be heavily frozen while still allowing new-language learning.

Default pilot:
- direction: EN -> ZH
- layers: 12, 20, 24
- freeze ratios inside the selected layer: 20%, 80%, 100%
- all other parameters stay trainable
- one full-FT baseline
- only 20% of ZH train blocks per condition

This keeps compute small: 10 conditions x 20% data ~= 2 full adaptation runs.

## Relation to prior work

HFT (ACL 2025) randomly fine-tunes only part of model parameters to reduce forgetting:
https://aclanthology.org/2025.acl-long.626/

SSU provides public code that also includes an HFT baseline:
https://github.com/gucci-j/ssu

This pilot does not import those codebases because the experiment here is different:
we freeze a random fraction only inside one selected layer, while all other layers remain trainable.
The point is to test a layer-specific mechanism hypothesis, not to claim random freezing itself is new.

## Run

```bash
git checkout zijian_test
git pull
bash experiments/random_layer_freeze_pilot/run_small_pilot.sh
```

Cheaper smoke test:

```bash
TRAIN_FRACTION=0.10 EVAL_MAX_BLOCKS=64 \
bash experiments/random_layer_freeze_pilot/run_small_pilot.sh
```

## Outputs

```text
random_layer_freeze_runs/en_to_zh_seed0/results.csv
random_layer_freeze_runs/en_to_zh_seed0/results_partial.csv
random_layer_freeze_runs/en_to_zh_seed0/manifest.json
random_layer_freeze_pilot.log
```

## Metrics

- forgetting = old-language post loss - old-language anchor loss
- new_language_gain = new-language anchor loss - new-language post loss
- forgetting_reduction_vs_full_ft
- plasticity_cost_vs_full_ft

## Interpretation

Interesting result:
- e.g. layer 12 + 80% freeze gives much lower forgetting with only a small loss in new-language gain,
- while the same freeze ratio at layers 20/24 is weaker or much more costly.

That would support layer-specific sparse plasticity.

If 80% freezing helps similarly at all layers, the effect is more consistent with generic selective fine-tuning/HFT rather than a special mechanism at the selected layer.

## Implementation detail

Layer numbering is 1-based:
- layer 12 -> model.model.layers[11]
- layer 20 -> model.model.layers[19]
- layer 24 -> model.model.layers[23]

For partial freezing, a fixed random element-wise mask zeros gradients at frozen positions.
After every AdamW step, those entries are restored exactly to their pre-adaptation values so decoupled weight decay cannot move them.
