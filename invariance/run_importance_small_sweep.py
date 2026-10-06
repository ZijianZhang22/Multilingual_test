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


def add_relative_metrics(df):
    baseline = df[df["method"] == "full_ft"]
    if baseline.empty:
        return df
    b = baseline.iloc[0]
    f0 = float(b["forgetting_loss_delta"])
    g0 = float(b["new_language_gain"])

    df = df.copy()
    if abs(f0) > 1e-12:
        df["forgetting_reduction_vs_full_ft"] = (
            f0 - df["forgetting_loss_delta"]
        ) / f0
    else:
        df["forgetting_reduction_vs_full_ft"] = float("nan")

    if abs(g0) > 1e-12:
        df["plasticity_cost_vs_full_ft"] = (
            g0 - df["new_language_gain"]
        ) / g0
    else:
        df["plasticity_cost_vs_full_ft"] = float("nan")
    return df


def mark_pareto(df):
    """Non-dominated points: lower forgetting and higher gain are better."""
    out = df.copy()
    pareto = []
    for i, row in out.iterrows():
        dominated = False
        for j, other in out.iterrows():
            if i == j:
                continue
            no_worse = (
                other["forgetting_loss_delta"] <= row["forgetting_loss_delta"]
                and other["new_language_gain"] >= row["new_language_gain"]
            )
            strictly_better = (
                other["forgetting_loss_delta"] < row["forgetting_loss_delta"]
                or other["new_language_gain"] > row["new_language_gain"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        pareto.append(not dominated)
    out["pareto_nondominated"] = pareto
    return out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Cheap one-direction exploratory sweep for Importance + Align. "
            "Runs a clean Full-FT baseline, a preservation-only lambda sweep, "
            "then a small alignment sweep at one chosen preservation lambda."
        )
    )
    ap.add_argument("--direction", default="zh:en", help="OLD:NEW")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument(
        "--aligned_data_file",
        default="invariance_analysis/causal_representation_suite/xnli_aligned.jsonl",
    )
    ap.add_argument(
        "--transferable_subspace_file",
        default="invariance_analysis/causal_representation_suite/transferable_rank64.pt",
    )
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--max_train_blocks", type=int, default=400)
    ap.add_argument("--preserve_lambdas", nargs="+", type=float, default=[0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--align_lambdas", nargs="+", type=float, default=[0.3, 1.0, 3.0])
    ap.add_argument("--alignment_preserve_lambda", type=float, default=1.0)
    ap.add_argument("--importance_batches", type=int, default=4)
    ap.add_argument("--importance_batch", type=int, default=2)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument(
        "--eval_languages",
        nargs="+",
        default=None,
        help="Defaults to OLD and NEW only for a cheap exploratory run.",
    )
    ap.add_argument(
        "--out_root",
        default="invariance_runs/small_importance_sweep",
    )
    ap.add_argument("--skip_align_sweep", action="store_true")
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

    # 0) Clean Full-FT baseline on exactly the same truncated/shuffled blocks.
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

    # 1) Preservation-only sweep: lambda_align = 0.
    for lam_p in args.preserve_lambdas:
        out_dir = root / f"preserve_p{tag(lam_p)}_a0"
        metrics = out_dir / "metrics.csv"
        cmd = [
            py, inv / "train_importance_preserve_align.py",
            *common,
            "--layer", args.layer,
            "--aligned_data_file", args.aligned_data_file,
            "--lambda_preserve", lam_p,
            "--lambda_align", 0.0,
            "--importance_batches", args.importance_batches,
            "--importance_batch", args.importance_batch,
            "--out_dir", out_dir,
        ]
        if Path(args.transferable_subspace_file).exists():
            cmd += ["--transferable_subspace_file", args.transferable_subspace_file]
        maybe_run(metrics, cmd, force=args.force)
        metric_files.append(metrics)

    # 2) Alignment sweep at one fixed preservation lambda.
    if not args.skip_align_sweep:
        for lam_a in args.align_lambdas:
            out_dir = root / (
                f"combined_p{tag(args.alignment_preserve_lambda)}_a{tag(lam_a)}"
            )
            metrics = out_dir / "metrics.csv"
            cmd = [
                py, inv / "train_importance_preserve_align.py",
                *common,
                "--layer", args.layer,
                "--aligned_data_file", args.aligned_data_file,
                "--lambda_preserve", args.alignment_preserve_lambda,
                "--lambda_align", lam_a,
                "--importance_batches", args.importance_batches,
                "--importance_batch", args.importance_batch,
                "--out_dir", out_dir,
            ]
            if Path(args.transferable_subspace_file).exists():
                cmd += ["--transferable_subspace_file", args.transferable_subspace_file]
            maybe_run(metrics, cmd, force=args.force)
            metric_files.append(metrics)

    frames = []
    for path in metric_files:
        if Path(path).exists():
            df = pd.read_csv(path)
            df["source_file"] = str(path)
            frames.append(df)

    if not frames:
        raise RuntimeError("No metrics were produced")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = add_relative_metrics(merged)
    merged = mark_pareto(merged)

    summary = root / "small_sweep_summary.csv"
    merged.to_csv(summary, index=False)

    print("\n=== Small importance/alignment sweep ===")
    cols = [
        c for c in [
            "method",
            "lambda_preserve",
            "lambda_align",
            "forgetting_loss_delta",
            "new_language_gain",
            "forgetting_reduction_vs_full_ft",
            "plasticity_cost_vs_full_ft",
            "pareto_nondominated",
        ]
        if c in merged.columns
    ]
    print(merged[cols].to_string(index=False))
    print(f"\nSaved: {summary}")
    print(
        "\nInterpretation: first inspect the preservation-only row family. "
        "Only if stronger preservation gives a useful retention/plasticity trade-off "
        "should the alignment sweep be expanded or repeated with more seeds."
    )


if __name__ == "__main__":
    main()
