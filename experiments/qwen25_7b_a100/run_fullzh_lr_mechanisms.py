#!/usr/bin/env python3
"""One-click 7B 1M-ZH causal mechanism comparison: LR=4e-5 vs LR=2e-5.

Reuse the existing LR-sweep EN anchor and sweep configuration. On demand,
retrain/save only the selected adapted checkpoint, fit 7B-specific subspaces,
run Step 7 (and optionally Step 6). Each LR has independent output files.

Does NOT silently use the original 2e-5 / 20%-ZH main runner.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SWEEP = HERE / "sweep_learning_rate.py"
FIT = ROOT / "experiments/retention_subspace_replication/build_core_subspaces.py"
STEP7 = ROOT / "experiments/retention_subspace_mechanism/run_step7_drift_isr_partition_rescue.py"
STEP6 = ROOT / "experiments/retention_subspace_mechanism/run_step6_energy_matched_controls.py"


def label(lr):
    return f"{lr:.8g}".replace("+", "")


def checkpoint_ok(path):
    # Verify every sharded weights file from the HF safetensors index.
    path = Path(path)
    if not (path / "config.json").is_file():
        return False
    idx = path / "model.safetensors.index.json"
    if idx.is_file():
        try:
            shards = set(json.loads(idx.read_text())["weight_map"].values())
            return bool(shards) and all(
                (path / name).is_file() and (path / name).stat().st_size > 0
                for name in shards
            )
        except (OSError, KeyError, ValueError, TypeError):
            return False
    file = path / "model.safetensors"
    return file.is_file() and file.stat().st_size > 0


def launch(cmd, dry_run=False):
    cmd = [str(x) for x in cmd]
    print("\n>>> " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def read_sweep_config(sweep_root, seed, default_anchor):
    cfg_file = sweep_root / f"seed{seed}" / "sweep_config.json"
    if cfg_file.is_file():
        cfg = json.loads(cfg_file.read_text())
        print(f"[sweep] existing config: {cfg_file}", flush=True)
    else:
        cfg = {
            "model_name": "Qwen/Qwen2.5-7B",
            "data_dir": "invariance_data/wiki",
            "out_root": str(sweep_root),
            "seed": seed,
            "old_language": "en",
            "new_language": "zh",
            "lrs": [2e-5, 4e-5, 6e-5],
            "anchor_lr": 2e-5,
            "anchor_checkpoint": str(default_anchor),
            "new_train_fraction": 1.0,
            "optimizer": "adamw8bit",
            "weight_decay": 0.1,
            "micro_batch": 1,
            "grad_accum": 16,
            "eval_batch": 1,
            "eval_max_blocks": 128,
            "max_optimizer_steps": 0,
            "save_checkpoints": False,
            "log_every": 10,
        }
        print("[sweep] no prior manifest; using full-ZH default config", flush=True)
    if int(cfg.get("seed", -1)) != seed:
        raise ValueError("Seed mismatch in existing sweep config.")
    if cfg.get("model_name") != "Qwen/Qwen2.5-7B":
        raise ValueError("This script is specifically for Qwen2.5-7B.")
    if cfg.get("old_language") != "en" or cfg.get("new_language") != "zh":
        raise ValueError("This comparison requires EN->ZH.")
    if abs(float(cfg.get("new_train_fraction", 0)) - 1.0) > 1e-12:
        raise ValueError("This comparison requires 100% ZH training blocks.")
    if int(cfg.get("max_optimizer_steps", 0)) != 0:
        raise ValueError("Refusing smoke-test checkpoints with max_optimizer_steps > 0.")
    if not cfg.get("anchor_checkpoint"):
        raise ValueError("Expected external EN anchor: set it when first running LR sweep.")
    return cfg


def build_sweep_command(cfg, lr, sweep_root):
    c = [
        sys.executable, SWEEP,
        "--model_name", cfg["model_name"],
        "--data_dir", cfg["data_dir"],
        "--out_root", cfg["out_root"],
        "--seed", cfg["seed"],
        "--old_language", cfg["old_language"],
        "--new_language", cfg["new_language"],
        "--lrs", *cfg["lrs"],
        "--anchor_lr", cfg.get("anchor_lr", 2e-5),
        "--anchor_checkpoint", cfg["anchor_checkpoint"],
        "--new_train_fraction", cfg["new_train_fraction"],
        "--optimizer", cfg.get("optimizer", "adamw8bit"),
        "--weight_decay", cfg.get("weight_decay", 0.1),
        "--micro_batch", cfg.get("micro_batch", 1),
        "--grad_accum", cfg.get("grad_accum", 16),
        "--eval_batch", cfg.get("eval_batch", 1),
        "--eval_max_blocks", cfg.get("eval_max_blocks", 128),
        "--max_optimizer_steps", cfg.get("max_optimizer_steps", 0),
        "--log_every", cfg.get("log_every", 10),
        "--export_lr", lr,
    ]
    if cfg.get("save_checkpoints", False):
        c.append("--save_checkpoints")
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_lrs", nargs="+", type=float, default=[4e-5, 2e-5])
    ap.add_argument("--sweep_root", default="replication_runs/qwen25_7b_a100_lr_fullzh")
    ap.add_argument("--anchor_checkpoint",
                    default="replication_runs/qwen25_7b_a100_lr_sweep/seed0/anchor",
                    help="Used only when creating a NEW full-ZH LR sweep config.")
    ap.add_argument("--analysis_root",
                    default="replication_runs/qwen25_7b_a100_fullzh_mechanisms")
    ap.add_argument("--through", choices=["step7", "step6"], default="step7")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--n_random", type=int, default=8)
    ap.add_argument("--extract_batch", type=int, default=1)
    ap.add_argument("--eval_batch", type=int, default=1)
    ap.add_argument("--eval_max_blocks", type=int, default=128)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if not args.target_lrs or len(args.target_lrs) != len(set(args.target_lrs)):
        ap.error("Provide one or more distinct target LRs.")
    if args.rank < 64 or args.layer < 1 or args.n_random < 2:
        ap.error("Need rank>=64, layer>=1, n_random>=2.")
    if min(args.eval_batch, args.extract_batch) < 1:
        ap.error("Batch sizes must be positive.")

    sweep_root = (ROOT / args.sweep_root).resolve()
    analysis_root = (ROOT / args.analysis_root).resolve()
    default_anchor = (ROOT / args.anchor_checkpoint).resolve()
    cfg = read_sweep_config(sweep_root, args.seed, default_anchor)
    valid_lrs = [float(x) for x in cfg["lrs"]]
    for lr in args.target_lrs:
        if lr not in valid_lrs:
            ap.error(f"LR {lr:g} not in existing sweep grid {valid_lrs}")
    anchor = (ROOT / cfg["anchor_checkpoint"]).resolve()
    if not args.dry_run and not checkpoint_ok(anchor):
        raise FileNotFoundError(
            f"EN anchor not found: {anchor}. Transfer it from the earlier run first."
        )
    if not args.dry_run:
        for lang in ("en", "zh"):
            for split in ("train", "val"):
                file = ROOT / cfg["data_dir"] / f"{lang}_{split}.pt"
                if not file.is_file():
                    raise FileNotFoundError(f"Missing training/evaluation data: {file}")

    for lr in args.target_lrs:
        lr_tag = label(lr)
        adapted = sweep_root / f"seed{args.seed}" / f"lr_{lr_tag}" / "adapted"
        run_dir = analysis_root / f"seed{args.seed}" / f"lr_{lr_tag}"
        subdir = run_dir / "subspaces"
        subspace_file = subdir / "core_subspaces.pt"
        step7_dir = run_dir / "drift_isr_partition"
        step6_dir = run_dir / "energy_controls"

        manifest = {
            "seed": args.seed, "learning_rate": lr,
            "train_fraction": cfg["new_train_fraction"],
            "model_name": cfg["model_name"],
            "anchor_checkpoint": str(anchor),
            "adapted_checkpoint": str(adapted),
            "sweep_config": cfg,
            "layer": args.layer, "rank": args.rank,
            "n_random": args.n_random,
            "eval_batch": args.eval_batch, "eval_max_blocks": args.eval_max_blocks,
            "extract_batch": args.extract_batch,
        }
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest_file = run_dir / "mechanism_manifest.json"
            if manifest_file.is_file():
                if json.loads(manifest_file.read_text()) != manifest:
                    raise ValueError(f"Incompatible prior mechanism settings: {manifest_file}")
            else:
                manifest_file.write_text(json.dumps(manifest, indent=2))
        print(f"\n======= 7B EN->ZH | LR {lr:g} | 100% ZH =======", flush=True)

        if checkpoint_ok(adapted):
            print(f"[resume] reuse adapted checkpoint {adapted}", flush=True)
        else:
            launch(build_sweep_command(cfg, lr, sweep_root), args.dry_run)
            if not args.dry_run and not checkpoint_ok(adapted):
                raise RuntimeError(f"Training reported success but checkpoint incomplete: {adapted}")

        if not subspace_file.is_file():
            launch([
                sys.executable, FIT,
                "--anchor_checkpoint", anchor,
                "--adapted_checkpoint", adapted,
                "--languages", "en", "zh", "fr", "de", "es",
                "--layer", args.layer,
                "--rank", args.rank,
                "--extract_batch", args.extract_batch,
                "--out_dir", subdir,
            ], args.dry_run)
        else:
            print(f"[resume] reuse subspaces {subspace_file}", flush=True)

        step7_summary = step7_dir / "partition_rescue_summary.csv"
        if not step7_summary.is_file():
            launch([
                sys.executable, STEP7,
                "--anchor_checkpoint", anchor,
                "--adapted_checkpoint", adapted,
                "--subspace_file", subspace_file,
                "--data_dir", cfg["data_dir"],
                "--old_language", "en",
                "--new_language", "zh",
                "--ks", 16, 32,
                "--alphas", 0.25, 0.5, 1.0,
                "--n_random", args.n_random,
                "--random_seed", 9700 + 100 * args.seed,
                "--eval_max_blocks", args.eval_max_blocks,
                "--eval_batch", args.eval_batch,
                "--out_dir", step7_dir,
            ], args.dry_run)
        else:
            print(f"[resume] reuse Step7 {step7_summary}", flush=True)

        if args.through == "step6":
            step6_summary = step6_dir / "energy_matched_summary.csv"
            if not step6_summary.is_file():
                launch([
                    sys.executable, STEP6,
                    "--anchor_checkpoint", anchor,
                    "--adapted_checkpoint", adapted,
                    "--subspace_file", subspace_file,
                    "--data_dir", cfg["data_dir"],
                    "--old_language", "en",
                    "--new_language", "zh",
                    "--subspaces", "transfer", "drift", "isr_cov",
                    "isr_multiclass", "vicreg",
                    "--strengths", 0.25, 0.5, 1.0,
                    "--n_random", args.n_random,
                    "--random_seed", 7300 + 100 * args.seed,
                    "--eval_max_blocks", args.eval_max_blocks,
                    "--eval_batch", args.eval_batch,
                    "--out_dir", step6_dir,
                ], args.dry_run)
            else:
                print(f"[resume] reuse Step6 {step6_summary}", flush=True)

        print(f"[complete] LR {lr:g}: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
