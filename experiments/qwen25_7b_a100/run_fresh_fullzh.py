#!/usr/bin/env python3
"""From-scratch 7B A100 EN->ZH full-data experiment.

Prepare Wiki EN/ZH blocks if missing, train a *new* EN anchor once,
fine-tune 7B on 100% ZH with LR=4e-5, fit subspaces, run Step 7.
An optional LR=2e-5 comparison can be added later if disk permits.

No prior checkpoints or old Pod data required. Nothing is deleted.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA_SCRIPT = ROOT / "invariance/prepare_wikipedia_multilang.py"
SWEEP_SCRIPT = HERE / "sweep_learning_rate.py"
FIT_SCRIPT = ROOT / "experiments/retention_subspace_replication/build_core_subspaces.py"
STEP7_SCRIPT = ROOT / "experiments/retention_subspace_mechanism/run_step7_drift_isr_partition_rescue.py"
STEP6_SCRIPT = ROOT / "experiments/retention_subspace_mechanism/run_step6_energy_matched_controls.py"
STAGES = ("training", "subspaces", "step7", "step6")


def label(lr: float) -> str:
    return f"{lr:.8g}".replace("+", "")


def checkpoint_ok(path: Path) -> bool:
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
    sf = path / "model.safetensors"
    return sf.is_file() and sf.stat().st_size > 0


def launch(command, dry_run):
    command = [str(item) for item in command]
    print("\n>>> " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def ensure_data(args):
    data_dir = ROOT / args.data_dir
    required = [data_dir / f"{lang}_{kind}.pt"
                for lang in ("en", "zh") for kind in ("train", "val")]
    exists = [p.is_file() for p in required]
    if all(exists):
        print(f"[data] reuse existing blocks in {data_dir}", flush=True)
        return
    if any(exists):
        missing = [str(p) for p, present in zip(required, exists) if not present]
        raise RuntimeError(
            "Partial Wiki dataset found. Refusing to overwrite older blocks; "
            f"missing: {missing}. Restore them or use a fresh --data_dir."
        )
    cmd = [
        sys.executable, DATA_SCRIPT,
        "--model_name", "Qwen/Qwen2.5-0.5B",
        "--languages", "en", "zh",
        "--train_tokens", 1_000_000,
        "--val_tokens", 100_000,
        "--block_size", 512,
        "--seed", 1234,
        "--out_dir", args.data_dir,
    ]
    launch(cmd, args.dry_run)


def sweep_cmd(args, lr):
    return [
        sys.executable, SWEEP_SCRIPT,
        "--model_name", "Qwen/Qwen2.5-7B",
        "--data_dir", args.data_dir,
        "--out_root", args.sweep_root,
        "--seed", args.seed,
        "--old_language", "en",
        "--new_language", "zh",
        "--lrs", *args.lrs,
        "--anchor_lr", "2e-5",
        "--new_train_fraction", "1.0",
        "--optimizer", args.optimizer,
        "--weight_decay", "0.1",
        "--micro_batch", "1",
        "--grad_accum", "16",
        "--eval_batch", "1",
        "--eval_max_blocks", args.eval_max_blocks,
        "--max_optimizer_steps", "0",
        "--export_lr", lr,
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lrs", nargs="+", type=float, default=[4e-5],
                    help="Default 4e-5 only; optionally append 2e-5 later.")
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--sweep_root",
                    default="replication_runs/qwen25_7b_a100_fresh_fullzh")
    ap.add_argument("--analysis_root",
                    default="replication_runs/qwen25_7b_a100_fresh_fullzh_analysis")
    ap.add_argument("--through", choices=STAGES, default="step7")
    ap.add_argument("--optimizer", choices=["adamw8bit", "paged_adamw8bit"],
                    default="adamw8bit")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n_random", type=int, default=8)
    ap.add_argument("--eval_max_blocks", type=int, default=128)
    ap.add_argument("--extract_batch", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if not args.lrs or any(x <= 0 or x >= float("inf") for x in args.lrs):
        ap.error("All LRs must be positive and finite.")
    if len(set(args.lrs)) != len(args.lrs):
        ap.error("Duplicate LRs are not allowed.")
    if args.layer < 1 or args.layer > 28 or args.rank < 64:
        ap.error("For 7B, 1<=layer<=28 and rank>=64.")
    if args.n_random < 2 or args.eval_max_blocks < 1 or args.extract_batch < 1:
        ap.error("Invalid n_random/eval_max_blocks/extract_batch.")

    print(f"[fresh-7b] seed={args.seed} ZH_fraction=1.0 LRs={args.lrs} "
          f"through={args.through}", flush=True)
    if len(args.lrs) > 1:
        print(
            "[disk warning] Each retained adapted 7B checkpoint adds ~15GB. "
            "With only 60GB available, start with one LR, then extend "
            "after checking free disk.", flush=True
        )

    ensure_data(args)

    sweep_root = (ROOT / args.sweep_root).resolve()
    analysis_root = (ROOT / args.analysis_root).resolve()
    anchor = sweep_root / f"seed{args.seed}" / "anchor"

    if not args.dry_run:
        sweep_root.mkdir(parents=True, exist_ok=True)
        available = shutil.disk_usage(sweep_root).free / 2**30
        print(f"[disk] free={available:.1f} GiB before model download/train",
              flush=True)
        if available < 35:
            raise RuntimeError(
                "Less than 35 GiB free. This is insufficient headroom for "
                "a 7B model cache, EN anchor, and adapted checkpoint. "
                "Increase Pod disk or move caches before proceeding."
            )

    selected_stages = STAGES[:STAGES.index(args.through) + 1]
    for lr in args.lrs:
        tag = f"lr_{label(lr)}"
        adapted = sweep_root / f"seed{args.seed}" / tag / "adapted"
        out = analysis_root / f"seed{args.seed}" / tag
        subspaces = out / "subspaces" / "core_subspaces.pt"
        step7 = out / "drift_isr_partition" / "partition_rescue_summary.csv"
        step6 = out / "energy_controls" / "energy_matched_summary.csv"
        print(f"\n========== LR {lr:g}; 100% ZH ==========", flush=True)

        if checkpoint_ok(adapted):
            print(f"[resume] reuse trained checkpoint {adapted}", flush=True)
        else:
            if not args.dry_run and (adapted.exists() and any(adapted.iterdir())):
                raise RuntimeError(
                    f"Partial checkpoint found at {adapted}. "
                    "Verify it is incomplete, then delete only that directory."
                )
            launch(sweep_cmd(args, lr), args.dry_run)
        if args.through == "training":
            continue

        if not subspaces.is_file():
            launch([
                sys.executable, FIT_SCRIPT,
                "--anchor_checkpoint", anchor,
                "--adapted_checkpoint", adapted,
                "--languages", "en", "zh", "fr", "de", "es",
                "--layer", args.layer,
                "--rank", args.rank,
                "--extract_batch", args.extract_batch,
                "--out_dir", out / "subspaces",
            ], args.dry_run)
        else:
            print(f"[resume] subspaces ready: {subspaces}", flush=True)

        if "step7" in selected_stages and not step7.is_file():
            launch([
                sys.executable, STEP7_SCRIPT,
                "--anchor_checkpoint", anchor,
                "--adapted_checkpoint", adapted,
                "--subspace_file", subspaces,
                "--data_dir", args.data_dir,
                "--old_language", "en",
                "--new_language", "zh",
                "--ks", 16, 32,
                "--alphas", 0.25, 0.5, 1.0,
                "--n_random", args.n_random,
                "--random_seed", 9700 + 100 * args.seed,
                "--eval_max_blocks", args.eval_max_blocks,
                "--eval_batch", 1,
                "--out_dir", out / "drift_isr_partition",
            ], args.dry_run)

        if args.through == "step6" and not step6.is_file():
            launch([
                sys.executable, STEP6_SCRIPT,
                "--anchor_checkpoint", anchor,
                "--adapted_checkpoint", adapted,
                "--subspace_file", subspaces,
                "--data_dir", args.data_dir,
                "--old_language", "en",
                "--new_language", "zh",
                "--subspaces", "transfer", "drift", "isr_cov",
                "isr_multiclass", "vicreg",
                "--strengths", 0.25, 0.5, 1.0,
                "--n_random", args.n_random,
                "--random_seed", 7300 + 100 * args.seed,
                "--eval_max_blocks", args.eval_max_blocks,
                "--eval_batch", 1,
                "--out_dir", out / "energy_controls",
            ], args.dry_run)
        print(f"[done] LR {lr:g}; results: {out}", flush=True)
    print(f"\nALL REQUESTED STAGES COMPLETE. Root: {analysis_root}", flush=True)


if __name__ == "__main__":
    main()
