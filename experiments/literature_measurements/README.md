## Starting with NO checkpoints (train from the public Qwen base)

The from-scratch runner now makes the necessary EN Anchor and ZH Adapted
checkpoints before measuring Drift / Transfer / ISR. This is sequential
**full fine-tuning**, using the repository's existing
`train_anchor_adapt.py`, not LoRA.

Run on a high-memory CUDA GPU with enough persistent storage:

```bash
cd /workspace/Multilingual_test
git fetch origin
git checkout feature/literature-measurement-suite
git pull --ff-only

bash experiments/literature_measurements/run_from_scratch.sh \
  --model-name Qwen/Qwen2.5-3B \
  --out /workspace/literature_3b_seed0 \
  --mode pilot
```

This sequence prepares EN/ZH Wikipedia training and validation tensors
(`OUT/wiki`), downloads the public Qwen base from Hugging Face, trains
EN then ZH with AdamW (20% of the prepared ZH training blocks by default),
saves `OUT/training/anchor` and `OUT/training/adapted`, evaluates
EN forgetting/ZH gain, and runs the prior pilot workflow under
`OUT/analysis`. It packages `OUT_results.tar.gz` (without model weights).

For an inexpensive correctness smoke test, run 0.5B in a NEW output folder:

```bash
bash experiments/literature_measurements/run_from_scratch.sh \
  --model-name Qwen/Qwen2.5-0.5B \
  --out /workspace/literature_smoke_0p5 \
  --train-tokens 131072 --val-tokens 16384 --block-size 256
```

The smoke setting is **not** a paper-comparable forgetting experiment;
the default 3B setup likewise requires replication before making claims.
Run all additional literature analyses after the pilot with `--mode full`
in a **new output directory**, or launch `run_all.sh` on the saved
checkpoints. `--dry-run` previews the bootstrap and analysis commands.

**Resource warning:** full 3B AdamW fine-tuning can exceed a 48 GB GPU.
An A100 80 GB-class GPU is safer. The script does not implicitly switch
models, parameter-efficient tuning or optimizer algorithms after OOM.

**Storage warning:** `OUT/training/anchor` and
`OUT/training/adapted` are NOT included in the default archive;
keep the complete `OUT/training` folder on persistent storage before
shutting down RunPod.

**Resume warning:** dataset preparation is skipped if verified outputs
remain. If training stops during ZH adaptation, the inherited two-stage
trainer will rerun the EN and ZH stages rather than resume an optimizer
state. Preserve sufficient GPU time for this first full run.

---
## One-command runner (recommended)

Once the code is on your RunPod, **one command** runs the selected workflow
from a fresh output folder or resumes already completed stages:

```bash
cd /workspace/Multilingual_test
git fetch origin
git checkout feature/literature-measurement-suite
git pull --ff-only

ANCHOR=/actual/path/to/stage1_en ADAPTED=/actual/path/to/stage2_zh \
  bash experiments/literature_measurements/run_all.sh
```

Defaults to **pilot**, containing clean core fit, held-out aligned XNLI,
semantic donor patching, bidirectional EN/ZH Wiki rescue and summary.
For **all experiments in this suite** (affine, layer similarity, aligned CKA,
transfer probe, whole-layer patch, mean-centered retrieval and association):

```bash
ANCHOR=/actual/path/to/stage1_en ADAPTED=/actual/path/to/stage2_zh MODE=full \
  bash experiments/literature_measurements/run_all.sh
```

Set `OUT=/workspace/results/my_3b_seed0` to make an independent run folder.
Optional environment overrides: `CORE_LAYER=20`, `LAYERS=6,12,20,24`,
`DATA_DIR=invariance_data/wiki`. Additional flags are accepted:

```bash
# Preview commands (no GPU required):
ANCHOR=/path/A ADAPTED=/path/B MODE=full \
  bash experiments/literature_measurements/run_all.sh --dry-run

# Resuming is automatic. Force reexecution:
ANCHOR=/path/A ADAPTED=/path/B \
  bash experiments/literature_measurements/run_all.sh --no-resume

# Repackage results without launching experiments or loading models:
python -m experiments.literature_measurements.run_all \
  --out mechanism_runs/literature_measurements_3b --archive-only

# Optional larger archive including extracted features:
python -m experiments.literature_measurements.run_all \
  --out mechanism_runs/literature_measurements_3b --archive-only --include-pt
```

**Outputs:** `OUT/logs/*.log`, `OUT/.completed/*.done`,
`OUT/run_manifest.json`, and `OUT_results.tar.gz`.
The default archive includes CSV, JSON, logs, and documentation but **excludes
large `.pt` features, checkpoints and Hugging Face weights**. To preserve
the learned subspace bases and extracted features use `--include-pt`.

**Resume contract:** completed stages are skipped only when marker AND
required output files exist; a different configuration in the same output
folder fails rather than silently mixing results. A crashed stage is rerun
on the next invocation. If you update code after fitting core bases, use
a new `OUT` folder when changing fitting logic so old intermediates cannot
be misinterpreted.

The runner requires local Anchor/Adapted checkpoints and EN/ZH Wiki
validation blocks (it does NOT train models or download checkpoints).
It calls the dataset preparation scripts to fetch XNLI from Hugging Face
if input files were not already generated, so the server needs dataset
access or a populated HF cache. Run the **pilot** first and check that
EN forgetting is nonzero before spending time on the **full** suite.
A 3B GPU end-to-end execution has NOT been confirmed.

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


## Targeted semantic × forgetting suite: Drift / Transfer / ISR ONLY

This is an additional **pilot**, NOT already-produced GPU results. Supported
bases: `drift`, `transfer`, `isr_cov`, `isr_multiclass` plus one
rank-matched random control for each. No VICReg or generic INLP in these new
interventions (INLP is used only internally while fitting Transfer).

**Refit core bases before a held-out claim.** Earlier
`build_core_subspaces.py` used `probe_train+probe_test` features for Drift
and its center; this branch fits Drift and center on `probe_train` only.
This guard refuses old/unverified bundles unless `--allow_legacy_core`
is deliberately used, in which case outputs are exploratory.

```bash
cd /workspace/Multilingual_test
git checkout feature/literature-measurement-suite
git pull --ff-only
ANCHOR=/path/to/qwen25_3b_stage1_en
ADAPTED=/path/to/qwen25_3b_stage2_zh
OUT=mechanism_runs/literature_measurements_3b
mkdir -p "$OUT"
# A train-only core fit; may take substantial time to fit 5 bases.
python -m experiments.retention_subspace_replication.build_core_subspaces \
  --anchor_checkpoint "$ANCHOR" --adapted_checkpoint "$ADAPTED" \
  --layer 20 --rank 64 --pool last \
  --probe_train_per_lang 500 --probe_test_per_lang 200 \
  --languages en zh fr de es \
  --out_dir "$OUT/fit_last_layer20"

CORE="$OUT/fit_last_layer20/core_subspaces.pt"
# The aligned test dataset is HELD OUT, not used for fitting the core.
python invariance/prepare_aligned_xnli.py --languages en zh \
  --split test --n_examples 120 \
  --out_file "$OUT/semantic_aligned_test.jsonl"

# (1) Model-level causal semantic donor comparison at final prompt token.
python -m experiments.literature_measurements.semantic_subspace_patch \
  --checkpoint "$ADAPTED" --core_file "$CORE" \
  --eval_jsonl "$OUT/semantic_aligned_test.jsonl" \
  --spaces drift transfer isr_cov isr_multiclass \
  --languages en zh --n_targets_per_language 6 \
  --out_dir "$OUT/semantic_subspace"

# (2) Reverse-direction, same-input causal LM patching.
python -m experiments.literature_measurements.bidirectional_subspace \
  --anchor_checkpoint "$ANCHOR" --adapted_checkpoint "$ADAPTED" \
  --core_file "$CORE" --languages en zh --max_blocks 32 \
  --spaces drift transfer isr_cov isr_multiclass \
  --data_dir invariance_data/wiki --out_dir "$OUT/bidirectional"

# (3) One side-by-side descriptive table.
python -m experiments.literature_measurements.subspace_functional_summary \
  --semantic_csv "$OUT/semantic_subspace/semantic_patching_summary.csv" \
  --bidirectional_csv "$OUT/bidirectional/bidirectional_subspace_summary.csv" \
  --out_dir "$OUT/functional_summary"
```

Outputs:
- `semantic_patching_examples.csv`: same-translation donor vs unrelated
  same-label and different-label donors; target NLI option accuracy/NLL,
  donor-label margin, raw and matched injection magnitudes.
- `semantic_patching_summary.csv`: per-space per-condition means.
- `bidirectional_subspace_summary.csv`: old/new-language loss change,
  fraction of EN forgetting recovered, fraction of forgetting induced on
  Anchor, actual intervention RMS.
- `subspace_functional_summary.csv`: descriptive bridge across experiments.

**Interpret carefully:** The semantic donor patch is *last prompt-token*
NLI, whereas bidirectional rescue uses *every token* on held-out Wiki. They
share basis/layer but are NOT identical tasks or sites. Semantic matched-pair
donors control for translation; a same-label random pair controls for label
information. Neither automatically proves semantic circuits. Bidirectional
effects establish reversible function sensitivity, not the training-time
causal origin of forgetting. Natural energy and matched energy are both
reported; matching uses shrink-only per-rank perturbations.

For a faster first pilot, use `--spaces drift transfer isr_multiclass`
and low `--n_targets_per_language`. Always retain the matched random
controls. Run the 3B GPU pilot before interpreting scientific outcomes.

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
