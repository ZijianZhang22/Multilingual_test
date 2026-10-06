import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def main():
    ap = argparse.ArgumentParser(
        description="Run the continual-training -> feature extraction -> INLP subspace tracking -> forgetting analysis pipeline."
    )
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--data_file", default="invariance_data/xnli_probe.jsonl")
    ap.add_argument("--reference_features", default="invariance_features/base.pt")
    ap.add_argument("--sequences", nargs="+", default=["en,zh", "zh,en"])
    ap.add_argument("--eval_languages", nargs="+", default=["en", "zh", "fr"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--extract_batch_size", type=int, default=16)
    ap.add_argument("--inlp_iters", type=int, default=32)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    root = Path(__file__).resolve().parent.parent
    inv = root / "invariance"

    runs_dir = Path(f"invariance_runs/sequence_seed{args.seed}")
    features_dir = Path(f"invariance_features/sequence_seed{args.seed}")
    subspace_dir = Path(f"invariance_analysis/subspace_seed{args.seed}")
    forgetting_dir = Path(f"invariance_analysis/subspace_forgetting_seed{args.seed}")

    if not args.skip_train:
        cmd = [
            py, inv / "train_sequence.py",
            "--model_name", args.model_name,
            "--data_dir", args.data_dir,
            "--sequences", *args.sequences,
            "--eval_languages", *args.eval_languages,
            "--out_dir", runs_dir,
            "--seed", args.seed,
            "--lr", args.lr,
            "--weight_decay", args.weight_decay,
            "--micro_batch", args.micro_batch,
            "--grad_accum", args.grad_accum,
            "--eval_batch", args.eval_batch,
        ]
        if args.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        run(cmd)

    run([
        py, inv / "extract_sequence_checkpoint_features.py",
        "--runs_dir", runs_dir,
        "--data_file", args.data_file,
        "--out_dir", features_dir,
        "--layer", args.layer,
        "--batch_size", args.extract_batch_size,
    ])

    feature_files = sorted(
        p for p in features_dir.glob("*/*.pt")
        if p.name.startswith("stage")
    )
    if not feature_files:
        raise FileNotFoundError(
            f"No checkpoint feature files found under {features_dir}"
        )

    run([
        py, inv / "track_inlp_subspaces.py",
        "--reference_features", args.reference_features,
        "--features_files", *feature_files,
        "--out_dir", subspace_dir,
        "--layer", args.layer,
        "--inlp_iters", args.inlp_iters,
        "--seed", args.seed,
    ])

    run([
        py, inv / "analyze_subspace_forgetting.py",
        "--sequence_metrics", runs_dir / "all_sequence_metrics.csv",
        "--subspace_by_language", subspace_dir / "subspace_tracking_by_language.csv",
        "--out_dir", forgetting_dir,
    ])

    print("\n=== PIPELINE COMPLETE ===")
    print(f"Training metrics: {runs_dir / 'all_sequence_metrics.csv'}")
    print(f"Subspace tracking: {subspace_dir / 'subspace_tracking_summary.csv'}")
    print(f"Old-language forgetting: {forgetting_dir / 'old_language_forgetting.csv'}")
    print(f"Correlations: {forgetting_dir / 'correlations.csv'}")


if __name__ == "__main__":
    main()
