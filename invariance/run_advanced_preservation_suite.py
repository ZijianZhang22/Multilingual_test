import argparse
import csv
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
    ordered = sorted(languages)
    return seed + 1000 + ordered.index(new_language)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "One-click runner for the three stronger follow-up methods: "
            "(1) importance-weighted preservation + alignment, "
            "(2) gradient-aware selective protection, and "
            "(3) SAE feature-level preservation."
        )
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument(
        "--directions",
        nargs="+",
        default=["en:zh", "zh:en"],
        help="OLD:NEW directions.",
    )
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
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--importance_batches", type=int, default=8)
    ap.add_argument("--importance_batch", type=int, default=2)
    ap.add_argument("--importance_preserve_lambda", type=float, default=1.0)
    ap.add_argument("--align_lambda", type=float, default=1.0)
    ap.add_argument("--gradient_strength", type=float, default=0.75)
    ap.add_argument("--gradient_mode", choices=["coordinate", "subspace"], default="coordinate")
    ap.add_argument("--sae_lambda", type=float, default=1.0)
    ap.add_argument("--sae_dict_size", type=int, default=2048)
    ap.add_argument("--sae_top_k", type=int, default=256)
    ap.add_argument("--sae_epochs", type=int, default=10)
    ap.add_argument(
        "--out_root",
        default="invariance_runs/advanced_methods",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke-test mode: seed0 EN->ZH only, 200 train blocks, "
            "4 importance batches, 4 SAE epochs."
        ),
    )
    args = ap.parse_args()

    py = sys.executable
    inv = Path(__file__).resolve().parent
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = [0] if args.quick else args.seeds
    directions = ["en:zh"] if args.quick else args.directions
    max_blocks = 200 if args.quick else None
    importance_batches = 4 if args.quick else args.importance_batches
    sae_epochs = 4 if args.quick else args.sae_epochs

    all_metric_files = []

    for seed in seeds:
        for direction in directions:
            if ":" not in direction:
                raise ValueError(f"Bad direction: {direction}")
            old, new = direction.split(":", 1)
            anchor = (
                Path(f"invariance_runs/sequence_seed{seed}")
                / f"{old}__{new}"
                / f"stage1_{old}"
            )
            if not anchor.exists():
                raise FileNotFoundError(
                    f"Missing anchor {anchor}. Run train_sequence.py first."
                )

            dseed = shuffle_seed(seed, new)
            root = out_root / f"seed{seed}" / f"{old}_to_{new}"
            root.mkdir(parents=True, exist_ok=True)

            common = [
                "--anchor_checkpoint", anchor,
                "--data_dir", args.data_dir,
                "--old_language", old,
                "--new_language", new,
                "--eval_languages", "en", "fr", "zh",
                "--layer", args.layer,
                "--seed", seed,
                "--data_shuffle_seed", dseed,
                "--micro_batch", args.micro_batch,
                "--grad_accum", args.grad_accum,
                "--eval_batch", args.eval_batch,
            ]
            if max_blocks is not None:
                common += ["--max_train_blocks", max_blocks]

            # 1) Importance-weighted preservation + semantic alignment
            m1 = root / "importance_align"
            m1_metrics = m1 / "metrics.csv"
            cmd1 = [
                py, inv / "train_importance_preserve_align.py",
                *common,
                "--aligned_data_file", args.aligned_data_file,
                "--lambda_preserve", args.importance_preserve_lambda,
                "--lambda_align", args.align_lambda,
                "--importance_batches", importance_batches,
                "--importance_batch", args.importance_batch,
                "--out_dir", m1,
            ]
            if Path(args.transferable_subspace_file).exists():
                cmd1 += [
                    "--transferable_subspace_file",
                    args.transferable_subspace_file,
                ]
            maybe_run(m1_metrics, cmd1, force=args.force)
            all_metric_files.append(m1_metrics)

            # 2) Gradient-aware selective protection
            m2 = root / f"gradient_{args.gradient_mode}"
            m2_metrics = m2 / "metrics.csv"
            maybe_run(
                m2_metrics,
                [
                    py, inv / "train_gradient_selective_protection.py",
                    *common,
                    "--surgery_mode", args.gradient_mode,
                    "--strength", args.gradient_strength,
                    "--importance_batches", importance_batches,
                    "--importance_batch", args.importance_batch,
                    "--out_dir", m2,
                ],
                force=args.force,
            )
            all_metric_files.append(m2_metrics)

            # 3) SAE feature-level preservation
            m3 = root / "sae_feature"
            m3_metrics = m3 / "metrics.csv"
            maybe_run(
                m3_metrics,
                [
                    py, inv / "train_sae_feature_preservation.py",
                    *common,
                    "--lambda_preserve", args.sae_lambda,
                    "--dict_size", args.sae_dict_size,
                    "--top_k_features", args.sae_top_k,
                    "--sae_epochs", sae_epochs,
                    "--importance_batches", importance_batches,
                    "--importance_batch", args.importance_batch,
                    "--out_dir", m3,
                ],
                force=args.force,
            )
            all_metric_files.append(m3_metrics)

    frames = []
    for path in all_metric_files:
        if Path(path).exists():
            df = pd.read_csv(path)
            df["source_file"] = str(path)
            frames.append(df)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(out_root / "advanced_methods_summary.csv", index=False)
        print("\n=== Advanced methods summary ===")
        cols = [
            c for c in [
                "method",
                "seed",
                "old_language",
                "new_language",
                "forgetting_loss_delta",
                "new_language_gain",
            ]
            if c in merged.columns
        ]
        print(merged[cols].to_string(index=False))
        print(f"Saved: {out_root / 'advanced_methods_summary.csv'}")


if __name__ == "__main__":
    main()
