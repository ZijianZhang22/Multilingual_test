import argparse
import subprocess
import sys
from pathlib import Path


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


def main():
    ap = argparse.ArgumentParser(
        description=(
            "One-click RQ1 diagnostic: for one OLD->NEW continual adaptation, "
            "measure behavioral forgetting context, anchor-vs-post hidden drift, "
            "and frozen-vs-refit probe recovery. Designed to ask whether apparent "
            "forgetting is representational erasure or loss of accessibility."
        )
    )
    ap.add_argument("--direction", default="zh:en", help="OLD:NEW")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--probe_languages", nargs="+", default=["en", "zh"])
    ap.add_argument("--probe_train_per_lang", type=int, default=600)
    ap.add_argument("--probe_test_per_lang", type=int, default=600)
    ap.add_argument("--probe_epochs", type=int, default=200)
    ap.add_argument("--probe_proj_dim", type=int, default=64)
    ap.add_argument("--irm_lambda", type=float, default=0.0)
    ap.add_argument("--inlp_iters", type=int, default=32)
    ap.add_argument("--extract_batch", type=int, default=16)
    ap.add_argument(
        "--runs_root",
        default="invariance_runs",
    )
    ap.add_argument(
        "--out_root",
        default="invariance_analysis/rq1_diagnostics",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if ":" not in args.direction:
        raise ValueError("--direction must be OLD:NEW, e.g. zh:en")
    old, new = args.direction.split(":", 1)

    inv = Path(__file__).resolve().parent
    py = sys.executable
    seq_dir = Path(args.runs_root) / f"sequence_seed{args.seed}" / f"{old}__{new}"
    anchor = seq_dir / f"stage1_{old}"
    post = seq_dir / f"stage2_{new}"
    if not anchor.exists() or not post.exists():
        raise FileNotFoundError(
            f"Need both {anchor} and {post}. Run train_sequence.py for {old},{new} first."
        )

    root = Path(args.out_root) / f"seed{args.seed}" / f"{old}_to_{new}"
    root.mkdir(parents=True, exist_ok=True)

    data_file = root / "xnli_probe.jsonl"
    anchor_features = root / "anchor_features.pt"
    post_features = root / "post_features.pt"
    probe_dir = root / "anchor_probe"
    probe_file = probe_dir / f"probe_layer_{args.layer}.pt"
    frozen_refit = root / "frozen_vs_refit.csv"
    subspace_file = root / "anchor_inlp_language_subspace.pt"
    drift_file = root / "anchor_post_drift.csv"

    maybe_run(
        data_file,
        [
            py, inv / "prepare_xnli.py",
            "--languages", *args.probe_languages,
            "--train_per_lang", args.probe_train_per_lang,
            "--test_per_lang", args.probe_test_per_lang,
            "--seed", 2026 + args.seed,
            "--out_file", data_file,
        ],
        force=args.force,
    )

    maybe_run(
        anchor_features,
        [
            py, inv / "extract_hidden.py",
            "--checkpoint", anchor,
            "--data_file", data_file,
            "--out_file", anchor_features,
            "--layers", args.layer,
            "--batch_size", args.extract_batch,
        ],
        force=args.force,
    )
    maybe_run(
        post_features,
        [
            py, inv / "extract_hidden.py",
            "--checkpoint", post,
            "--data_file", data_file,
            "--out_file", post_features,
            "--layers", args.layer,
            "--batch_size", args.extract_batch,
        ],
        force=args.force,
    )

    maybe_run(
        probe_file,
        [
            py, inv / "fit_invariant_probe.py",
            "--features_file", anchor_features,
            "--out_dir", probe_dir,
            "--proj_dim", args.probe_proj_dim,
            "--irm_lambda", args.irm_lambda,
            "--epochs", args.probe_epochs,
            "--seed", args.seed,
        ],
        force=args.force,
    )

    maybe_run(
        frozen_refit,
        [
            py, inv / "validate_frozen_vs_refit.py",
            "--reference_probe", probe_file,
            "--features_files", anchor_features, post_features,
            "--out_file", frozen_refit,
            "--seed", args.seed,
        ],
        force=args.force,
    )

    maybe_run(
        subspace_file,
        [
            py, inv / "fit_inlp_language_subspace.py",
            "--features_file", anchor_features,
            "--out_file", subspace_file,
            "--layer", args.layer,
            "--iters", args.inlp_iters,
            "--seed", args.seed,
        ],
        force=args.force,
    )

    maybe_run(
        drift_file,
        [
            py, inv / "analyze_anchor_post_drift.py",
            "--anchor_features", anchor_features,
            "--post_features", post_features,
            "--language_subspace_file", subspace_file,
            "--layer", args.layer,
            "--out_file", drift_file,
        ],
        force=args.force,
    )

    print("\n=== RQ1 DIAGNOSTIC COMPLETE ===")
    print(f"Direction: {old.upper()} -> {new.upper()}")
    print(f"Frozen/refit detail: {frozen_refit}")
    print(f"Frozen/refit summary: {frozen_refit.with_name(frozen_refit.stem + '_summary.csv')}")
    print(f"Anchor/post drift: {drift_file}")
    print("\nInterpretation:")
    print(
        "1) If frozen probe drops after adaptation but refit probe recovers, "
        "the task information may remain but its geometry/accessibility changed."
    )
    print(
        "2) If both frozen and refit probes drop, that is more consistent with "
        "loss of linearly recoverable task information."
    )
    print(
        "3) Compare old-language drift in the INLP language subspace versus the "
        "residual/shared complement; do not treat two-language seed0 drift as "
        "correlational evidence yet."
    )


if __name__ == "__main__":
    main()
