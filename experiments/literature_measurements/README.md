# Literature-inspired measurements of multilingual forgetting

**Branch:** `feature/literature-measurement-suite` (based on `zijian_test`).
This is a **new evaluation suite**, not an implementation of a new training method.
All reported measurements must be interpreted alongside **English forgetting**
and **Chinese adaptation gain** on held-out Wiki validation blocks.

## What it measures (and what it does not)

| Module | Research precedent | Measurement | Evidence limit |
|---|---|---|---|
| `extract_attention.py` + `layer_similarity.py` | LayerMoE (ACL 2025) | Layerwise all-pairs cross-language token cosine, plus a global-mean-centered control | Similarity, *not* causal shared functionality |
| `layer_similarity.py` | First Align, then Predict (EACL 2021) | Pair-ID-aligned linear CKA | CKA requires matching examples; not proof of causal alignment |
| `affine_projection.py` | The Geometry of Multilingual Language Model Representations (EMNLP 2022) | Same-language, cross-language, and mean-recentered cross-language affine PCA interventions, PPL ratio | **Decoder-only causal-LM adaptation**, not the source masked-LM evaluation |
| `retrieval.py` | On the Language Neutrality (Findings EMNLP 2020) | Raw, global-mean- and language-mean-centered held-out aligned-pair retrieval R@1 | Decodability/alignment, not causal usage |\n| `transfer_probe.py` | Universal Grammatical Relations (ACL 2020) | In-language, direct, leave-one-language-out, joint XNLI task probes | **Task-label transfer**; not a replication of their structural/syntactic probe |
| `layer_patch.py` | Layer intervention / causal tracing literature | Full-layer Anchor->Adapted activation patching; optional layer weight restoration | The effective intervention site need not be the original site of damage |
| `association.py` | Our research question | Join per-layer similarity changes and EN patching improvements | Correlations are exploratory, *not* independent causal evidence |

Research question: **Does cross-lingual sharedness and its adaptation-induced
change predict the old-language loss recovered by a layer intervention?**

## Prerequisites

Run from the **repository root** (the imports are rooted at
`experiments.literature_measurements`). Install this repository's existing
`torch`, `transformers`, `datasets`, and `accelerate` dependencies.
A CUDA GPU is needed for Qwen2.5-3B checkpoint evaluations. Avoid 7B until
a clear 3B pilot exists. The suite assumes identical tokenizer/model family
between Stage-1 Anchor and Stage-2 Adapted.

**Expected data formats:**

- `invariance/prepare_xnli.py` creates rows with `language`, `split`
  (`probe_train` from XNLI validation, `probe_test` from XNLI test),
  `label`, `premise`, `hypothesis`, and `example_id`.
- `invariance/prepare_aligned_xnli.py` creates cross-language aligned
  `pair_id` rows with `split="aligned"`, from a selected validation or
  test partition. Use **test** for held-out paired representation analysis.
- `affine_projection.py` accepts JSONL containing `text` or
  `premise`+`hypothesis`; provide separate **fit** and **eval** JSONLs.
  It rejects overlapping example IDs and identical input texts.
- For LM causal tests: `invariance_data/wiki/en_val.pt` and
  `invariance_data/wiki/zh_val.pt` containing `input_ids`.

## One-model 3B pilot

Set your **actual** checkpoints; these are placeholders, not expected repo
paths:

```bash
cd /workspace/Multilingual_test
git checkout feature/literature-measurement-suite
ANCHOR=/path/to/qwen25_3b_stage1_en
ADAPTED=/path/to/qwen25_3b_stage2_zh
OUT=mechanism_runs/literature_measurements_3b
mkdir -p "$OUT"

python invariance/prepare_xnli.py --languages en zh fr de es \
  --train_per_lang 600 --test_per_lang 600 \
  --out_file "$OUT/xnli_probe.jsonl"
python invariance/prepare_aligned_xnli.py --split test \
  --languages en zh fr de es --n_examples 200 \
  --out_file "$OUT/xnli_aligned_test.jsonl"
```

### 1. LayerMoE-style attention-output cosine (hold-out XNLI)

```bash
for name in anchor adapted; do
  if [ "$name" = anchor ]; then CKPT="$ANCHOR"; else CKPT="$ADAPTED"; fi
  python -m experiments.literature_measurements.extract_attention \
    --checkpoint "$CKPT" --data_file "$OUT/xnli_probe.jsonl" \
    --out_file "$OUT/${name}_attention.pt" --split probe_test \
    --languages en zh fr de es --source attention \
    --layers 1,6,12,18,24 --pool tokens
done
python -m experiments.literature_measurements.layer_similarity \
  --anchor_features "$OUT/anchor_attention.pt" \
  --adapted_features "$OUT/adapted_attention.pt" \
  --out_csv "$OUT/layer_similarity.csv" --split probe_test --languages en zh
```

**Interpretation:** `source=attention` hooks the output of
`decoder.layers[l-1].self_attn`. This deliberately differs from the residual
stream used by existing Step 4/6/7 rescue; it is **not** legitimate to compare
them as if they came from one intervention site.

Optional aligned-pair CKA (attention output, mean-pooled over valid tokens):

```bash
for name in anchor adapted; do
  if [ "$name" = anchor ]; then CKPT="$ANCHOR"; else CKPT="$ADAPTED"; fi
  python -m experiments.literature_measurements.extract_attention \
    --checkpoint "$CKPT" --data_file "$OUT/xnli_aligned_test.jsonl" \
    --out_file "$OUT/${name}_aligned_attention.pt" --split aligned \
    --languages en zh --source attention --pool mean --layers 1,6,12,18,24
done
python -m experiments.literature_measurements.layer_similarity \
  --anchor_features "$OUT/anchor_aligned_attention.pt" \
  --adapted_features "$OUT/adapted_aligned_attention.pt" \
  --out_csv "$OUT/layer_aligned_cka.csv" --split aligned --languages en zh
```

Note that CKA of unrelated, unpaired English/Chinese tokens is **not**
valid aligned-sample CKA. The script leaves the field blank when no paired
`pair_id` metadata exists.

### 2. Chang et al.-inspired affine subspace projection (LM-level)

**Fit and evaluation must use disjoint samples and identical projection
sites.** For a simple pilot, split `xnli_probe.jsonl` by its `split`
field into `xnli_fit.jsonl` and `xnli_eval.jsonl`:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("mechanism_runs/literature_measurements_3b")
rows = [json.loads(s) for s in (p/"xnli_probe.jsonl").read_text().splitlines() if s]
for source, target in (("probe_train","xnli_fit.jsonl"),("probe_test","xnli_eval.jsonl")):
    (p/target).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows if r["split"]==source))
PY
python -m experiments.literature_measurements.affine_projection \
  --anchor_checkpoint "$ANCHOR" --adapted_checkpoint "$ADAPTED" \
  --fit_jsonl "$OUT/xnli_fit.jsonl" --eval_jsonl "$OUT/xnli_eval.jsonl" \
  --languages en zh --layers 12,20,24 --max_rank 256 \
  --out_dir "$OUT/affine"
```

Results: `affine/affine_basis_metadata.csv`,
`affine/affine_projection_results.csv` and fitting basis `.pt` files.
`target_reached=false` means the limited rank did **not** reach 90%
variance; **never call it a 90%-explained-variance projection**. The same
language-specific basis is fitted separately for Anchor and Adapted.

**Important:** the experimental loss here is *causal next-token NLL on XNLI
texts*. It is not an NLI *answer accuracy*. We additionally report
`ppl_ratio=exp(intervention_nll - baseline_nll)`.

### 3. Cross-lingual task transfer probe on existing residual features

```bash
for name in anchor adapted; do
  if [ "$name" = anchor ]; then CKPT="$ANCHOR"; else CKPT="$ADAPTED"; fi
  python invariance/extract_hidden.py --checkpoint "$CKPT" \
    --data_file "$OUT/xnli_probe.jsonl" \
    --layers 12 20 24 --pool last \
    --out_file "$OUT/${name}_residual.pt"
done
python -m experiments.literature_measurements.transfer_probe \
  --anchor_features "$OUT/anchor_residual.pt" \
  --adapted_features "$OUT/adapted_residual.pt" \
  --languages en zh fr de es --layers 12,20,24 --rank 64 \
  --out_csv "$OUT/transfer_probe.csv"
```

The default comparisons (`anchor_pca`, `drift_train`, `random`) are
fitted on `probe_train` only. Use `--basis_file` to evaluate the repository's
existing INLP/Transfer/ISR/VICReg bundle **only after auditing that its bases
were trained without test features**, and only for its matching layer.
All classifiers are fixed-budget ridge probes. Retrieval and language ID
can additionally be assessed with existing `semantic_causal_validation`
scripts; they are **not** substitutes for model-level behavior.

### 4. Full-layer anchor activation / parameter restoration

```bash
python -m experiments.literature_measurements.layer_patch \
  --anchor_checkpoint "$ANCHOR" --adapted_checkpoint "$ADAPTED" \
  --data_dir invariance_data/wiki --old_language en --languages en zh \
  --layers 1,6,12,18,24 --eval_max_blocks 64 \
  --out_csv "$OUT/layer_patch.csv"
# Add --weight_restore for more expensive per-layer weight restoration.
```

Connect **only comparable layers** to the HSA cosine curve:

```bash
python -m experiments.literature_measurements.association \
  --similarity_csv "$OUT/layer_similarity.csv" \
  --patch_csv "$OUT/layer_patch.csv" \
  --out_dir "$OUT/association"
```

This evaluates whether *attention-output language similarity* correlates
with *full residual layer replacement*. They are different measurement sites;
correlations are exploratory. For a like-for-like site comparison, extract
`--source residual` and repeat the layer similarity step.

### 5. Language Neutrality: held-out retrieval after mean centering

Fit language means on an **aligned validation** set and evaluate retrieval on a
separate **aligned test** set. Do not reuse the aligned test examples for
subspace fitting or mean estimation.

```bash
python invariance/prepare_aligned_xnli.py --split validation \
  --languages en zh fr --n_examples 300 \
  --out_file "$OUT/xnli_aligned_fit.jsonl"
# The aligned_test file was created earlier in the workflow.
for name in anchor adapted; do
  if [ "$name" = anchor ]; then CKPT="$ANCHOR"; else CKPT="$ADAPTED"; fi
  for part in fit test; do
    if [ "$part" = fit ]; then INPUT="$OUT/xnli_aligned_fit.jsonl"; else INPUT="$OUT/xnli_aligned_test.jsonl"; fi
    python -m experiments.literature_measurements.extract_attention \
      --checkpoint "$CKPT" --data_file "$INPUT" \
      --out_file "$OUT/${name}_aligned_${part}_residual.pt" \
      --split aligned --languages en zh fr --source residual \
      --pool mean --layers 12,20,24 --max_rows_per_language 250
  done
done
python -m experiments.literature_measurements.retrieval \
  --anchor_fit "$OUT/anchor_aligned_fit_residual.pt" \
  --anchor_eval "$OUT/anchor_aligned_test_residual.pt" \
  --adapted_fit "$OUT/adapted_aligned_fit_residual.pt" \
  --adapted_eval "$OUT/adapted_aligned_test_residual.pt" \
  --languages en zh fr --out_csv "$OUT/mean_centered_retrieval.csv"
```

Outputs directional R@1 on unseen aligned pairs for raw vectors, global-mean
centering and language-specific centering. Centers use **fit pairs only**;
the script checks pair-ID separation. The score is retrieval/representation
alignment, not proof the model *uses* these vectors in its computations.

## Experimental hygiene

1. **Separate fitting, model selection, and reporting**: do not fit bases,
   train probes, or choose interventions on `probe_test` data. If comparing
   many layers on the test set, repeat the selected hypothesis on a fresh test
   set.
2. **Use held-out aligned pairs for CKA**; unpaired all-pairs cosine and
   matched-pair CKA answer different questions.
3. **Compare natural and matched energy** for projected interventions; rank
   equality alone is not sufficient. Existing Step 6/7 control protocols
   remain authoritative for drift/random comparisons.
4. **Report EN and ZH together**: restoring old-language performance might
   damage the newly learned language.
5. **Do not infer universal layer semantics** from 0.5B/1.5B/3B: all are Qwen
   family. Use seed replications and compare training budget / LR / pooling.

## Lightweight tests

```bash
python -m compileall -q experiments/literature_measurements \
  experiments/retention_subspace_replication/build_core_subspaces.py
python -m unittest discover -s experiments/literature_measurements/tests -v
```

These are CPU-only numerical tests; they do not verify 3B inference or
checkpoint-dependent conclusions. The end-to-end 3B pilot must be run
separately on GPU.

## References

- Zhang et al., *Less, but Better: Efficient Multilingual Expansion for LLMs
  via Layer-wise Mixture-of-Experts*, ACL 2025:
  https://aclanthology.org/2025.acl-long.878/
- Muller et al., *First Align, then Predict*, EACL 2021:
  https://aclanthology.org/2021.eacl-main.189/
- Chang et al., *The Geometry of Multilingual Language Model
  Representations*, EMNLP 2022:
  https://aclanthology.org/2022.emnlp-main.9/
- Chi, Hewitt and Manning, *Finding Universal Grammatical Relations in
  Multilingual BERT*, ACL 2020:
  https://aclanthology.org/2020.acl-main.493/
- Libovický, Rosa and Fraser, *On the Language Neutrality of Pre-trained
  Multilingual Representations*, Findings of EMNLP 2020:
  https://aclanthology.org/2020.findings-emnlp.150/
