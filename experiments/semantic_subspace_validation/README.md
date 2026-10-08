# Semantic Subspace Validation — Qwen2.5-7B

This is the independent validation module for our existing EN->ZH full-FT
7B experiment. **It reuses the saved EN Anchor, ZH Adapted, and core_subspaces.pt.**
It does not retrain either checkpoint or rerun Step 6/7.

## Setup

From Multilingual_test repository root:

```bash
git checkout feature/qwen25-7b-a100-8bit
git pull origin feature/qwen25-7b-a100-8bit
pip install scikit-learn
python -m unittest discover -s experiments/semantic_subspace_validation -p 'test_suite.py' -v
```

## One-click 7B validation

```bash
nohup python -u experiments/semantic_subspace_validation/run_existing_7b.py \
 --through causal --n_examples 18 \
 > semantic_validation_7b.log 2>&1 < /dev/null &
tail -f semantic_validation_7b.log
```

Run only probing + retrieval first by using `--through probes`; rerun
with `--through causal` later to reuse prior results.

The pre-existing paths default to:
`replication_runs/qwen25_7b_a100_fresh_fullzh/seed0/anchor`,
`replication_runs/qwen25_7b_a100_fresh_fullzh/seed0/lr_4e-05/adapted`,
and `replication_runs/qwen25_7b_a100_fresh_fullzh_analysis/seed0/lr_4e-05/subspaces/`.

Results:
`replication_runs/qwen25_7b_a100_semantic_validation/seed0/lr_4e-05/`,
containing `probes_and_retrieval/probe_results.csv`,
`retrieval_results.csv`, `causal_semantics/causal_summary.csv`,
`ablation_example_level.csv`, `swap_example_level.csv`,
`REPORT.md`, and `evidence_report.json`.

## Methods and caveats

- Basis comparisons: ISR, Drift, Transfer, VICReg, PCA, rank-matched random,
  Drift top/bottom 16/32 ordered by ISR projection.
- Linear NLI and language-ID probes trained on XNLI validation, evaluated
  on XNLI test; leave-language-out means the **probe** not the fitted basis.
- Cross-lingual aligned XNLI **test** retrieval, English-pivot lexical hard
  negatives with the same NLI label.
- Frozen-model last-token ablation and donor swaps (natural and norm-matched).
  This is **not** proof of semantic equivalence: bases were fitted using
  mean-pooled hidden states; zero-shot NLI prompt quality also matters.
- Step 7 from the existing run is used for comparison only.

GPU end-to-end validation has not yet been performed in this GitHub branch.
