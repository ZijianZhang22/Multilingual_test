#!/usr/bin/env python3
"""Single-A100 7B learning-rate / forgetting sweep (no Step 6/7).

Train EN anchor once; repeat identical ZH adaptation from that checkpoint
with several learning rates. Uses the same 8-bit full-parameter optimizer
and data-selection protocol as train_7b_a100.py.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from types import SimpleNamespace


def positive_float(value):
    number = float(value)
    if not (0 < number < float("inf")):
        raise argparse.ArgumentTypeError("learning rate must be positive and finite")
    return number


def canonical_lr(value):
    return f"{value:.8g}".replace("+", "")


def safe_write_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def verify_or_extend_config(config_path: Path, config: dict):
    """Allow new LR values to be appended without retraining the same anchor.

    Every non-LR hyperparameter must be identical. Previously scheduled LRs
    must remain present, so existing metrics stay part of the same experiment.
    """
    if not config_path.is_file():
        safe_write_json(config_path, config)
        return
    previous = json.loads(config_path.read_text())
    previous_lrs = previous.get("lrs", [])
    other_changed = [
        key for key in set(previous) | set(config)
        if key != "lrs" and previous.get(key) != config.get(key)
    ]
    if other_changed or not set(previous_lrs).issubset(set(config["lrs"])):
        raise ValueError(
            "Existing sweep uses incompatible settings; "
            f"changed={sorted(other_changed)}; choose a new --out_root."
        )
    if previous_lrs != config["lrs"]:
        safe_write_json(config_path, config)
        print(f"[resume] extended LR grid: {previous_lrs} -> {config['lrs']}", flush=True)


def save_table(out_dir: Path, lrs):
    rows = []
    for lr in lrs:
        path = out_dir / f"lr_{canonical_lr(lr)}" / "metrics.json"
        if path.is_file():
            rows.append(json.loads(path.read_text()))
    if not rows:
        return
    fields = [
        "learning_rate", "seed", "old_language", "new_language",
        "anchor_en_loss", "adapted_en_loss", "forgetting_loss_delta",
        "anchor_zh_loss", "adapted_zh_loss", "new_language_gain",
        "zh_tokens_seen", "zh_optimizer_steps", "checkpoint_saved",
    ]
    tmp = out_dir / "lr_forgetting_summary.csv.tmp"
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out_dir / "lr_forgetting_summary.csv")
    print("\nLearning rate | EN forgetting | ZH gain", flush=True)
    for row in rows:
        print(f"{row['learning_rate']:.1e} | {row['forgetting_loss_delta']:+.6f} | "
              f"{row['new_language_gain']:+.6f}", flush=True)
    print(f"Saved: {out_dir/'lr_forgetting_summary.csv'}", flush=True)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--out_root", default="replication_runs/qwen25_7b_a100_lr_sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--lrs", nargs="+", type=positive_float,
                    default=[1e-5, 2e-5, 4e-5, 5e-5, 6e-5])
    ap.add_argument("--anchor_lr", type=positive_float, default=2e-5,
                    help="EN anchor training LR, fixed across the sweep")
    ap.add_argument("--anchor_checkpoint", default=None,
                    help="Optional existing EN anchor (skips EN training).")
    ap.add_argument("--new_train_fraction", type=float, default=0.2)
    ap.add_argument("--optimizer", choices=["adamw8bit", "paged_adamw8bit"],
                    default="adamw8bit")
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--eval_batch", type=int, default=1)
    ap.add_argument("--eval_max_blocks", type=int, default=128)
    ap.add_argument("--max_optimizer_steps", type=int, default=0,
                    help="0 is full epoch; positive values ONLY for smoke tests")
    ap.add_argument("--save_checkpoints", action="store_true",
                    help="Save every adapted checkpoint (~15 GB each). Off by default.")
    ap.add_argument("--export_lr", type=positive_float, default=None,
                    help="After sweep, retrain and save ONLY the selected LR checkpoint.")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--dry_run", action="store_true")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if not 0 < args.new_train_fraction <= 1:
        ap.error("--new_train_fraction must be within (0,1]")
    if args.export_lr is not None and args.export_lr not in args.lrs:
        ap.error("--export_lr must be one of --lrs")
    if len(set(args.lrs)) != len(args.lrs):
        ap.error("--lrs must not contain duplicate values")
    if min(args.micro_batch, args.grad_accum, args.eval_batch, args.log_every) < 1:
        ap.error("batch sizes and log_every must be positive")
    if args.max_optimizer_steps < 0:
        ap.error("--max_optimizer_steps must be >= 0")
    if args.eval_max_blocks < 0:
        ap.error("--eval_max_blocks must be >= 0")

    output = Path(args.out_root) / f"seed{args.seed}"
    config = {
        key: val for key, val in vars(args).items() if key not in {"dry_run", "export_lr"}
    }
    if args.dry_run:
        print(f"[dry-run] output={output}")
        print(f"[dry-run] train EN once lr={args.anchor_lr:g} unless anchor_checkpoint set")
        for lr in args.lrs:
            print(f"[dry-run] reload same EN anchor -> ZH lr={lr:g}, "
                  f"train_fraction={args.new_train_fraction:.2f}; "
                  f"save_checkpoint={args.save_checkpoints}")
        if args.export_lr is not None:
            print(f"[dry-run] export selected LR={args.export_lr:g} only")
        print("[dry-run] evaluate EN+ZH; write lr_forgetting_summary.csv")
        return

    import torch
    from train_7b_a100 import (
        complete_checkpoint, ensure_checkpoint_space, load_model,
        measure, memory_report, save_model, train_stage,
    )
    from transformers import AutoTokenizer
    import sys
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from invariance.train_sequence import load_blocks, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment requires an A100 80GB-class CUDA BF16 GPU.")
    if torch.cuda.get_device_properties(0).total_memory < 70 * 2**30:
        raise RuntimeError("Configured for approximately 80GB GPU RAM.")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "sweep_config.json"
    verify_or_extend_config(config_path, config)

    data = Path(args.data_dir)
    old_train = load_blocks(data / f"{args.old_language}_train.pt")
    new_all = load_blocks(data / f"{args.new_language}_train.pt")
    old_val = load_blocks(data / f"{args.old_language}_val.pt")
    new_val = load_blocks(data / f"{args.new_language}_val.pt")
    if args.eval_max_blocks:
        old_val, new_val = old_val[:args.eval_max_blocks], new_val[:args.eval_max_blocks]
    g_old = torch.Generator().manual_seed(args.seed + 1000)
    old_train = old_train[torch.randperm(len(old_train), generator=g_old)]
    g_new = torch.Generator().manual_seed(args.seed + 1001)
    new_all = new_all[torch.randperm(len(new_all), generator=g_new)]
    new_train = new_all[:max(1, round(len(new_all) * args.new_train_fraction))]
    print(f"[data] EN {len(old_train)} blocks, ZH {len(new_train)} blocks; "
          f"val {len(old_val)}/{len(new_val)} blocks", flush=True)

    anchor = Path(args.anchor_checkpoint).resolve() if args.anchor_checkpoint else output / "anchor"
    anchor_meta = output / "anchor_summary.json"
    if args.anchor_checkpoint:
        if not complete_checkpoint(anchor):
            raise FileNotFoundError(f"External anchor incomplete: {anchor}")
        print("[anchor] external checkpoint: verify that its training protocol "
              "matches your 7B experiment before using these results", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not args.anchor_checkpoint and not (complete_checkpoint(anchor) and anchor_meta.is_file()):
        if anchor.exists() and any(anchor.iterdir()):
            raise RuntimeError("Incomplete anchor exists; investigate before restarting: " + str(anchor))
        set_seed(args.seed)
        model = load_model(args.model_name)
        trainargs = SimpleNamespace(**vars(args))
        trainargs.lr = args.anchor_lr
        print(f"[anchor] EN training LR={args.anchor_lr:g}", flush=True)
        old_stats = train_stage(model, old_train, trainargs)
        losses = measure(model, old_val, new_val, args.eval_batch)
        save_model(model, tokenizer, anchor)
        safe_write_json(anchor_meta, {"old_train": old_stats,
                                      "old_loss": losses["old"],
                                      "new_loss": losses["new"],
                                      "anchor_lr": args.anchor_lr})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if not anchor_meta.is_file() or args.anchor_checkpoint:
        model = load_model(anchor)
        losses = measure(model, old_val, new_val, args.eval_batch)
        safe_write_json(anchor_meta, {"old_loss": losses["old"],
                                      "new_loss": losses["new"],
                                      "anchor_checkpoint": str(anchor)})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    reference = json.loads(anchor_meta.read_text())
    print(f"[anchor] EN={reference['old_loss']:.6f} "
          f"ZH={reference['new_loss']:.6f}", flush=True)

    for lr in args.lrs:
        if args.export_lr is not None and lr != args.export_lr:
            continue
        tag = f"lr_{canonical_lr(lr)}"
        result_dir = output / tag
        result_dir.mkdir(exist_ok=True)
        metric_path = result_dir / "metrics.json"
        wants_checkpoint = args.save_checkpoints or args.export_lr == lr
        checkpoint_dir = result_dir / "adapted"
        if metric_path.is_file() and (not wants_checkpoint or complete_checkpoint(checkpoint_dir)):
            print(f"[resume] skip {tag}", flush=True)
            save_table(output, args.lrs)
            continue
        if wants_checkpoint:
            if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                raise FileExistsError(
                    f"An incomplete or orphaned checkpoint exists: {checkpoint_dir}. "
                    "Verify it contains no usable complete checkpoint, then "
                    "delete ONLY that incomplete adapted/ directory before retrying."
                )
            ensure_checkpoint_space(anchor, checkpoint_dir)
        set_seed(args.seed)
        model = load_model(anchor)
        trainargs = SimpleNamespace(**vars(args))
        trainargs.lr = lr
        print(f"\n[adapt] {tag} ZH learning rate={lr:g}", flush=True)
        stats = train_stage(model, new_train, trainargs)
        losses = measure(model, old_val, new_val, args.eval_batch)
        checkpoint_saved = False
        row = {
            "learning_rate": lr,
            "seed": args.seed,
            "old_language": args.old_language,
            "new_language": args.new_language,
            "anchor_en_loss": reference["old_loss"],
            "adapted_en_loss": losses["old"],
            "forgetting_loss_delta": losses["old"] - reference["old_loss"],
            "anchor_zh_loss": reference["new_loss"],
            "adapted_zh_loss": losses["new"],
            "new_language_gain": reference["new_loss"] - losses["new"],
            "zh_tokens_seen": stats["tokens_seen"],
            "zh_optimizer_steps": stats["optimizer_steps"],
            "checkpoint_saved": checkpoint_saved,
        }
        # Preserve completed numerical results even if checkpoint save fails.
        safe_write_json(metric_path, row)
        if wants_checkpoint:
            save_model(model, tokenizer, checkpoint_dir)
            row["checkpoint_saved"] = True
            safe_write_json(metric_path, row)
        print(f"[result] LR={lr:.1e} forgetting={row['forgetting_loss_delta']:+.6f} "
              f"ZH_gain={row['new_language_gain']:+.6f}", flush=True)
        memory_report(f"{tag}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        save_table(output, args.lrs)


if __name__ == "__main__":
    main()
