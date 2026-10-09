#!/usr/bin/env python3
"""One-command EN->ZH training + Drift/Transfer/ISR study without checkpoints.

Two resume-safe bootstrap steps:
  (1) stream/tokenize multilingual Wikipedia into train/held-out val Wiki blocks;
  (2) fine-tune a public Qwen2.5 base on EN then ZH, saving stage checkpoints.
Then delegate to existing literature_measurements.run_all (pilot or full).

The training protocol reuses retention_subspace_replication/train_anchor_adapt.py.
This is full fine-tuning, NOT LoRA. Changes in model, seed, training budget
or tokenizer must use a different --out directory to preserve validity.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.literature_measurements.run_all import (
    ROOT, Step, command, make_archive, run_steps,
)

PYTHON = sys.executable


def build_bootstrap(a):
    out = Path(a.out).expanduser().resolve()
    data = Path(a.data_dir).expanduser().resolve() if a.data_dir else out / "wiki"
    trained = out / "training"
    anchor = trained / "anchor"
    adapted = trained / "adapted"
    langs = (a.old_language, a.new_language)
    preparation = Step(
        "00_prepare_wikipedia",
        [PYTHON, "invariance/prepare_wikipedia_multilang.py",
         "--model_name", a.model_name, "--languages", *langs,
         "--train_tokens", str(a.train_tokens),
         "--val_tokens", str(a.val_tokens), "--block_size", str(a.block_size),
         "--seed", str(a.data_seed), "--out_dir", str(data)],
        tuple(data / f"{lang}_{partition}.pt"
              for lang in langs for partition in ("train", "val")) +
        (data / "metadata.json",),
    )
    train_cmd = [
        PYTHON, "experiments/retention_subspace_replication/train_anchor_adapt.py",
        "--model_name", a.model_name, "--data_dir", str(data),
        "--old_language", a.old_language, "--new_language", a.new_language,
        "--seed", str(a.seed), "--lr", str(a.lr),
        "--weight_decay", str(a.weight_decay),
        "--micro_batch", str(a.micro_batch),
        "--grad_accum", str(a.grad_accum),
        "--eval_batch", str(a.eval_batch),
        "--eval_max_blocks", str(a.eval_max_blocks),
        "--new_train_fraction", str(a.new_train_fraction),
        "--out_dir", str(trained), "--reload_anchor_before_new_stage",
    ]
    if a.gradient_checkpointing:
        train_cmd.append("--gradient_checkpointing")
    training = Step("00_train_en_then_zh", train_cmd,
                    (anchor/"config.json", adapted/"config.json",
                     trained/"training_summary.json"))
    analysis_cmd = [
        PYTHON, "-m", "experiments.literature_measurements.run_all",
        "--anchor", str(anchor), "--adapted", str(adapted),
        "--out", str(out/"analysis"), "--mode", a.mode,
        "--old-language", a.old_language,
        "--new-language", a.new_language,
        "--fit-languages", *a.fit_languages,
        "--core-layer", str(a.core_layer),
        "--layers", a.layers,
        "--data-dir", str(data),
        "--batch-size", str(a.extract_batch),
        "--max-blocks", str(a.eval_max_blocks),
        "--rank", str(a.rank),
        "--train-per-lang", str(a.probe_train_per_lang),
        "--test-per-lang", str(a.probe_test_per_lang),
        "--aligned-fit-examples", str(a.aligned_fit_examples),
        "--aligned-test-examples", str(a.aligned_test_examples),
        "--targets-per-lang", str(a.targets_per_lang),
    ]
    return (preparation, training), analysis_cmd, data, anchor, adapted


def validate_data_metadata(path, a):
    manifest = path / "metadata.json"
    if not manifest.exists():
        if any(path.glob("*_train.pt")) or any(path.glob("*_val.pt")):
            raise RuntimeError(
                f"{path} contains partial data without metadata. "
                "Choose a fresh --out/--data-dir or clear incomplete files manually."
            )
        return
    m = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {
        "model_name": a.model_name,
        "block_size": a.block_size,
        "train_tokens_requested_per_language": a.train_tokens,
        "val_tokens_requested_per_language": a.val_tokens,
    }
    mismatch = {k: (m.get(k), v) for k, v in expected.items()
                if m.get(k) != v}
    if sorted(m.get("languages", [])) != sorted([a.old_language, a.new_language]):
        mismatch["languages"] = (m.get("languages"), [a.old_language, a.new_language])
    if mismatch:
        raise RuntimeError(
            f"Wikipedia data at {path} is not compatible: {mismatch}. "
            "Use an independent output folder, never reuse tokenized data from a different model."
        )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B",
                   help="Hub model ID; 0.5B recommended for a first smoke test")
    p.add_argument("--out", default="mechanism_runs/from_scratch_3b_seed0")
    p.add_argument("--data-dir", default="",
                   help="Existing or new wiki data folder. Default: OUT/wiki")
    p.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=1234)
    p.add_argument("--old-language", default="en")
    p.add_argument("--new-language", default="zh")
    p.add_argument("--fit-languages", nargs="+",
                   default=["en", "zh", "fr", "de", "es"])
    p.add_argument("--train-tokens", type=int, default=1_000_000,
                   help="Wikipedia train tokens PER language")
    p.add_argument("--val-tokens", type=int, default=100_000,
                   help="Wikipedia held-out validation tokens PER language")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--new-train-fraction", type=float, default=0.20,
                   help="ZH stage uses this fraction of its prepared train blocks")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--eval-batch", type=int, default=4)
    p.add_argument("--eval-max-blocks", type=int, default=64)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--core-layer", type=int, default=20)
    p.add_argument("--layers", default="6,12,20,24")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--extract-batch", type=int, default=2)
    p.add_argument("--probe-train-per-lang", type=int, default=350)
    p.add_argument("--probe-test-per-lang", type=int, default=150)
    p.add_argument("--aligned-fit-examples", type=int, default=200)
    p.add_argument("--aligned-test-examples", type=int, default=120)
    p.add_argument("--targets-per-lang", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--archive-only", action="store_true")
    p.add_argument("--include-pt", action="store_true")
    a = p.parse_args(argv)
    if a.archive_only:
        return a
    if a.old_language == a.new_language:
        p.error("old/new languages must differ")
    if not {a.old_language, a.new_language}.issubset(a.fit_languages):
        p.error("fit-languages must include old and new languages")
    if a.mode == "full" and a.core_layer not in {
            int(x) for x in a.layers.replace(",", " ").split()}:
        p.error("--layers must contain the core layer in full mode")
    if min(a.train_tokens, a.val_tokens) < a.block_size:
        p.error("token budgets must be at least one block")
    if not 0 < a.new_train_fraction <= 1:
        p.error("new-train-fraction must lie in (0,1]")
    if a.lr <= 0 or a.micro_batch < 1 or a.grad_accum < 1 or a.eval_batch < 1:
        p.error("learning-rate/batches must be positive")
    if a.eval_max_blocks < 1 or a.extract_batch < 1:
        p.error("eval-max-blocks/extract-batch must be positive")
    return a


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out).expanduser().resolve()
    if a.archive_only:
        print("Results archive:", make_archive(out, a.include_pt))
        return
    boot, analysis_cmd, data, anchor, adapted = build_bootstrap(a)
    if a.dry_run:
        for step in boot:
            print(f"[DRY RUN] {step.name}\n  {' '.join(step.argv)}")
        print(f"[DRY RUN] analysis\n  {' '.join(analysis_cmd)}")
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train Qwen. Use --dry-run to preview.")
    validate_data_metadata(data, a)
    # The original run_steps machinery records a strict manifest and
    # restarts unsuccessful stages. The TRAIN stage is all-or-nothing:
    # interruptions during ZH training will rerun both language stages.
    run_steps(a, list(boot))
    for ckpt in (anchor, adapted):
        if not (ckpt/"config.json").is_file():
            raise RuntimeError(f"Checkpoint missing after training: {ckpt}")
        if not (list(ckpt.glob("*.safetensors")) or
                list(ckpt.glob("pytorch_model*.bin"))):
            raise RuntimeError(f"Checkpoint weights missing: {ckpt}")
    summary = json.loads((out/"training"/"training_summary.json").read_text())
    forgetting = float(summary["forgetting"])
    gain = float(summary["new_language_gain"])
    print(f"[STAGE SUMMARY] EN forgetting={forgetting:+.6f}, "
          f"ZH gain={gain:+.6f}")
    if forgetting <= 0:
        print("WARNING: No positive EN forgetting found; restoration fraction "
              "is not interpretable as recovery. Consider a different predeclared "
              "training protocol rather than tuning on test data.", flush=True)
    # It is valid to rerun this analysis independently: nested runner
    # manages its own manifest and stages under OUT/analysis.
    subprocess.run(analysis_cmd, cwd=ROOT, check=True)
    final = make_archive(out, a.include_pt)
    print(f"Complete. Export: {final}")
    print(f"Checkpoints: {anchor} and {adapted}")
    print("Important: preserve OUT/training/** on persistent volume before stopping GPU.")


if __name__ == "__main__":
    main()
