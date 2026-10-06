import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    cmd = [str(x) for x in cmd]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def maybe_run(output_path, cmd, force=False):
    output_path = Path(output_path)
    if output_path.exists() and not force:
        print(f"SKIP existing: {output_path}")
        return
    run(cmd)


def direction_seed(old_lang, new_lang, seed, all_train_languages):
    ordered = sorted(all_train_languages)
    idx = ordered.index(new_lang)
    return seed + 1000 + idx


def main():
    ap = argparse.ArgumentParser(
        description=(
            "One-click causal multilingual representation suite: "
            "fit language/transferable subspaces, train both language orders, "
            "run dense preservation sweeps, reverse-direction tests, learning "
            "curves, manipulation checks, and Pareto aggregation."
        )
    )
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--xnli_probe", default="invariance_data/xnli_probe.jsonl")
    ap.add_argument("--base_features", default="invariance_features/base.pt")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--inlp_iters", type=int, default=32)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument(
        "--directions",
        nargs="+",
        default=["en:zh", "zh:en"],
        help="OLD:NEW directions, e.g. en:zh zh:en",
    )
    ap.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.3, 1.0, 3.0, 5.0, 10.0, 20.0],
    )
    ap.add_argument("--curve_lambda", type=float, default=3.0)
    ap.add_argument(
        "--curve_fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50, 0.75, 1.00],
    )
    ap.add_argument("--manipulation_lambda", type=float, default=10.0)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--extract_batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument(
        "--out_root",
        default="invariance_runs/causal_representation_suite",
    )
    ap.add_argument(
        "--analysis_root",
        default="invariance_analysis/causal_representation_suite",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Fast validation mode: seed 0, lambdas 1/10, "
            "curve fractions 0.25/0.5/1.0."
        ),
    )
    args = ap.parse_args()

    py = sys.executable
    inv = Path(__file__).resolve().parent
    out_root = Path(args.out_root)
    analysis_root = Path(args.analysis_root)
    out_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)

    seeds = [0] if args.quick else args.seeds
    lambdas = [1.0, 10.0] if args.quick else args.lambdas
    curve_fractions = (
        [0.25, 0.50, 1.0] if args.quick else args.curve_fractions
    )

    parsed_dirs = []
    for item in args.directions:
        if ":" not in item:
            raise ValueError(f"Bad direction '{item}', expected OLD:NEW")
        old, new = item.split(":", 1)
        parsed_dirs.append((old, new))
    train_languages = sorted({x for pair in parsed_dirs for x in pair})

    # ------------------------------------------------------------------
    # 1) Fit the reusable INLP language subspace from base Layer-12 features.
    # ------------------------------------------------------------------
    lang_subspace = analysis_root / "reference_inlp_language_subspace.pt"
    maybe_run(
        lang_subspace,
        [
            py, inv / "fit_inlp_language_subspace.py",
            "--features_file", args.base_features,
            "--out_file", lang_subspace,
            "--layer", args.layer,
            "--iters", args.inlp_iters,
            "--seed", 0,
        ],
        force=args.force,
    )

    # ------------------------------------------------------------------
    # 2) Build aligned parallel XNLI and fit a matched-rank transferable basis.
    # ------------------------------------------------------------------
    aligned_jsonl = analysis_root / "xnli_aligned.jsonl"
    maybe_run(
        aligned_jsonl,
        [
            py, inv / "prepare_aligned_xnli.py",
            "--languages", *train_languages, "fr",
            "--split", "validation",
            "--n_examples", 2000,
            "--out_file", aligned_jsonl,
        ],
        force=args.force,
    )

    aligned_features = analysis_root / "xnli_aligned_base_features.pt"
    maybe_run(
        aligned_features,
        [
            py, inv / "extract_hidden.py",
            "--checkpoint", args.model_name,
            "--data_file", aligned_jsonl,
            "--out_file", aligned_features,
            "--layers", args.layer,
            "--batch_size", args.extract_batch,
        ],
        force=args.force,
    )

    transfer_subspace = analysis_root / f"transferable_rank{args.rank}.pt"
    maybe_run(
        transfer_subspace,
        [
            py, inv / "fit_transferable_subspace.py",
            "--features_file", aligned_features,
            "--aligned_data_file", aligned_jsonl,
            "--language_subspace_file", lang_subspace,
            "--out_file", transfer_subspace,
            "--layer", args.layer,
            "--rank", args.rank,
        ],
        force=args.force,
    )

    methods = ["full_ft", "lang", "shared64", "transfer64", "shared", "full"]
    sweep_metrics = []

    for seed in seeds:
        # --------------------------------------------------------------
        # 3) Ensure seed-specific stage-1 anchor checkpoints exist.
        # --------------------------------------------------------------
        sequence_dir = Path(f"invariance_runs/sequence_seed{seed}")
        sequence_metrics = sequence_dir / "all_sequence_metrics.csv"
        seq_args = [
            f"{old},{new}" for old, new in parsed_dirs
        ]
        maybe_run(
            sequence_metrics,
            [
                py, inv / "train_sequence.py",
                "--model_name", args.model_name,
                "--data_dir", args.data_dir,
                "--sequences", *seq_args,
                "--eval_languages", *sorted(set(train_languages + ["fr"])),
                "--out_dir", sequence_dir,
                "--seed", seed,
                "--lr", args.lr,
                "--weight_decay", args.weight_decay,
                "--micro_batch", args.micro_batch,
                "--grad_accum", args.grad_accum,
                "--eval_batch", args.eval_batch,
            ],
            force=args.force,
        )

        for old, new in parsed_dirs:
            branch = f"{old}__{new}"
            anchor = sequence_dir / branch / f"stage1_{old}"
            if not anchor.exists():
                raise FileNotFoundError(f"Missing anchor checkpoint: {anchor}")

            shuffle_seed = direction_seed(old, new, seed, train_languages)
            direction_root = out_root / f"seed{seed}" / f"{old}_to_{new}"
            direction_root.mkdir(parents=True, exist_ok=True)

            # ----------------------------------------------------------
            # 4) Dense lambda sweep = retention/plasticity Pareto data.
            # Save only lambda chosen for the later manipulation check.
            # ----------------------------------------------------------
            sweep_dir = direction_root / "sweep"
            sweep_csv = sweep_dir / "intervention_metrics.csv"
            maybe_run(
                sweep_csv,
                [
                    py, inv / "train_representation_preservation.py",
                    "--anchor_checkpoint", anchor,
                    "--subspace_file", lang_subspace,
                    "--transferable_subspace_file", transfer_subspace,
                    "--data_dir", args.data_dir,
                    "--old_language", old,
                    "--new_language", new,
                    "--eval_languages", *sorted(set(train_languages + ["fr"])),
                    "--methods", *methods,
                    "--lambdas", *lambdas,
                    "--layer", args.layer,
                    "--shared64_rank", args.rank,
                    "--seed", seed,
                    "--data_shuffle_seed", shuffle_seed,
                    "--lr", args.lr,
                    "--weight_decay", args.weight_decay,
                    "--micro_batch", args.micro_batch,
                    "--grad_accum", args.grad_accum,
                    "--eval_batch", args.eval_batch,
                    "--out_dir", sweep_dir,
                    "--feature_extract_data", args.xnli_probe,
                    "--feature_extract_dir",
                    analysis_root / f"seed{seed}" / f"{old}_to_{new}" / "manipulation" / "features",
                    "--feature_extract_methods", *methods,
                    "--feature_extract_lambdas", args.manipulation_lambda,
                    "--feature_extract_batch", args.extract_batch,
                ],
                force=args.force,
            )
            sweep_metrics.append(sweep_csv)

            # ----------------------------------------------------------
            # 5) Learning curves at one representative lambda.
            # ----------------------------------------------------------
            curve_dir = direction_root / f"curves_lam{args.curve_lambda:g}"
            curve_csv = curve_dir / "learning_curves.csv"
            maybe_run(
                curve_csv,
                [
                    py, inv / "train_representation_preservation.py",
                    "--anchor_checkpoint", anchor,
                    "--subspace_file", lang_subspace,
                    "--transferable_subspace_file", transfer_subspace,
                    "--data_dir", args.data_dir,
                    "--old_language", old,
                    "--new_language", new,
                    "--eval_languages", *sorted(set(train_languages + ["fr"])),
                    "--methods", *methods,
                    "--lambdas", args.curve_lambda,
                    "--curve_fractions", *curve_fractions,
                    "--layer", args.layer,
                    "--shared64_rank", args.rank,
                    "--seed", seed,
                    "--data_shuffle_seed", shuffle_seed,
                    "--lr", args.lr,
                    "--weight_decay", args.weight_decay,
                    "--micro_batch", args.micro_batch,
                    "--grad_accum", args.grad_accum,
                    "--eval_batch", args.eval_batch,
                    "--out_dir", curve_dir,
                ],
                force=args.force,
            )

            # ----------------------------------------------------------
            # 6) Manipulation check using features extracted in memory during
            # the sweep. This avoids writing ~1 GB model checkpoints for every
            # intervention solely for representation analysis.
            # ----------------------------------------------------------
            manip_dir = (
                analysis_root / f"seed{seed}" / f"{old}_to_{new}" / "manipulation"
            )
            features_dir = manip_dir / "features"
            features_dir.mkdir(parents=True, exist_ok=True)

            anchor_features = features_dir / "anchor.pt"
            maybe_run(
                anchor_features,
                [
                    py, inv / "extract_hidden.py",
                    "--checkpoint", anchor,
                    "--data_file", args.xnli_probe,
                    "--out_file", anchor_features,
                    "--layers", args.layer,
                    "--batch_size", args.extract_batch,
                ],
                force=args.force,
            )

            intervention_features = []
            for method in methods:
                if method == "full_ft":
                    feat = features_dir / "full_ft.pt"
                else:
                    feat = (
                        features_dir
                        / f"{method}_lam{args.manipulation_lambda:g}.pt"
                    )
                if feat.exists():
                    intervention_features.append(feat)
                else:
                    print(f"WARNING: manipulation feature missing: {feat}")

            drift_csv = manip_dir / "intervention_drift.csv"
            if intervention_features:
                maybe_run(
                    drift_csv,
                    [
                        py, inv / "analyze_intervention_drift.py",
                        "--anchor_features", anchor_features,
                        "--intervention_features", *intervention_features,
                        "--language_subspace_file", lang_subspace,
                        "--transferable_subspace_file", transfer_subspace,
                        "--out_file", drift_csv,
                        "--layer", args.layer,
                    ],
                    force=args.force,
                )

    # ------------------------------------------------------------------
    # 7) Aggregate seeds + directions into the retention/plasticity frontier.
    # ------------------------------------------------------------------
    pareto_dir = analysis_root / "pareto_all"
    pareto_csv = pareto_dir / "aggregate_pareto.csv"
    maybe_run(
        pareto_csv,
        [
            py, inv / "analyze_pareto.py",
            "--inputs", *sweep_metrics,
            "--out_dir", pareto_dir,
        ],
        force=args.force,
    )

    print("\n=== CAUSAL REPRESENTATION SUITE COMPLETE ===")
    print(f"Language subspace: {lang_subspace}")
    print(f"Transferable subspace: {transfer_subspace}")
    print(f"Pareto summary: {pareto_csv}")
    print(f"Runs root: {out_root}")
    print(f"Analysis root: {analysis_root}")


if __name__ == "__main__":
    main()
