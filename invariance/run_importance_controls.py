import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(cmd):
    cmd = [str(x) for x in cmd]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def maybe_run(output, cmd, force=False):
    output = Path(output)
    if output.exists() and not force:
        print(f"SKIP existing: {output}")
        return
    run(cmd)


def shuffle_seed(seed, new_language, languages=("en", "zh")):
    ordered = sorted(set(languages) | {new_language})
    return seed + 1000 + ordered.index(new_language)


def tag(x):
    return str(x).replace("-", "m").replace(".", "p")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Matched control study for importance-weighted representation preservation. "
            "Compares importance vs uniform vs shuffled coordinate weights at the same "
            "lambda, data, anchor, optimizer, and training blocks."
        )
    )
    ap.add_argument("--direction", default="zh:en", help="OLD:NEW")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--max_train_blocks", type=int, default=400)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[3.0, 5.0])
    ap.add_argument(
        "--controls",
        nargs="+",
        choices=["importance", "uniform", "shuffled"],
        default=["importance", "uniform", "shuffled"],
    )
    ap.add_argument("--importance_batches", type=int, default=4)
    ap.add_argument("--importance_batch", type=int, default=2)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--weight_shuffle_seed", type=int, default=2026)
    ap.add_argument(
        "--eval_languages",
        nargs="+",
        default=None,
        help="Defaults to OLD and NEW only.",
    )
    ap.add_argument(
        "--out_root",
        default="invariance_runs/importance_controls",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if ":" not in args.direction:
        raise ValueError("--direction must be OLD:NEW, e.g. zh:en")
    old, new = args.direction.split(":", 1)
    eval_languages = args.eval_languages or [old, new]

    inv = Path(__file__).resolve().parent
    py = sys.executable

    anchor = (
        Path(f"invariance_runs/sequence_seed{args.seed}")
        / f"{old}__{new}"
        / f"stage1_{old}"
    )
    if not anchor.exists():
        raise FileNotFoundError(
            f"Missing anchor {anchor}. Run train_sequence.py for {old},{new} first."
        )

    dseed = shuffle_seed(args.seed, new, languages=(old, new))
    root = (
        Path(args.out_root)
        / f"seed{args.seed}"
        / f"{old}_to_{new}"
        / f"blocks{args.max_train_blocks}"
    )
    root.mkdir(parents=True, exist_ok=True)

    common = [
        "--anchor_checkpoint", anchor,
        "--data_dir", args.data_dir,
        "--old_language", old,
        "--new_language", new,
        "--eval_languages", *eval_languages,
        "--seed", args.seed,
        "--data_shuffle_seed", dseed,
        "--max_train_blocks", args.max_train_blocks,
        "--micro_batch", args.micro_batch,
        "--grad_accum", args.grad_accum,
        "--eval_batch", args.eval_batch,
        "--lr", args.lr,
        "--weight_decay", args.weight_decay,
    ]

    metric_files = []

    # Matched Full-FT baseline on exactly the same truncated/shuffled new-language blocks.
    baseline_dir = root / "full_ft"
    baseline_metrics = baseline_dir / "metrics.csv"
    maybe_run(
        baseline_metrics,
        [
            py, inv / "train_anchor_full_ft.py",
            *common,
            "--out_dir", baseline_dir,
        ],
        force=args.force,
    )
    metric_files.append(baseline_metrics)

    for lam in args.lambdas:
        for control in args.controls:
            out_dir = root / f"{control}_lambda{tag(lam)}"
            metrics = out_dir / "metrics.csv"
            maybe_run(
                metrics,
                [
                    py, inv / "train_importance_preserve_align.py",
                    *common,
                    "--layer", args.layer,
                    "--lambda_preserve", lam,
                    "--lambda_align", 0.0,
                    "--importance_batches", args.importance_batches,
                    "--importance_batch", args.importance_batch,
                    "--weight_control", control,
                    "--weight_shuffle_seed", args.weight_shuffle_seed,
                    "--out_dir", out_dir,
                ],
                force=args.force,
            )
            metric_files.append(metrics)

    frames = []
    for path in metric_files:
        if Path(path).exists():
            df = pd.read_csv(path)
            df["source_file"] = str(path)
            frames.append(df)

    if not frames:
        raise RuntimeError("No metrics produced")

    merged = pd.concat(frames, ignore_index=True, sort=False)

    baseline = merged[merged["method"] == "full_ft"]
    if baseline.empty:
        raise RuntimeError("Missing Full-FT baseline metrics")
    b = baseline.iloc[0]
    f0 = float(b["forgetting_loss_delta"])
    g0 = float(b["new_language_gain"])

    merged["forgetting_reduction_vs_full_ft"] = (
        (f0 - merged["forgetting_loss_delta"]) / f0
        if abs(f0) > 1e-12 else float("nan")
    )
    merged["plasticity_cost_vs_full_ft"] = (
        (g0 - merged["new_language_gain"]) / g0
        if abs(g0) > 1e-12 else float("nan")
    )

    summary_path = root / "importance_control_summary.csv"
    merged.to_csv(summary_path, index=False)

    print("\n=== Importance-weight controls ===")
    cols = [
        c for c in [
            "method",
            "weight_control",
            "lambda_preserve",
            "forgetting_loss_delta",
            "new_language_gain",
            "forgetting_reduction_vs_full_ft",
            "plasticity_cost_vs_full_ft",
        ]
        if c in merged.columns
    ]
    print(merged[cols].to_string(index=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
