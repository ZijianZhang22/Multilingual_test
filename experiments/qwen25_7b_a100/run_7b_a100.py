#!/usr/bin/env python3
"""Reproducible Qwen2.5-7B single-A100 Step 6/7 runner.

Stages: BF16 full-FT via 8-bit optimizer; five subspaces; Step7 then Step6.
All derived subspaces and interventions reuse the existing repository scripts.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STAGES = ["training", "subspaces", "step7", "step6"]


def checkpoint_ok(path):
    path = Path(path)
    return (path / "config.json").is_file() and bool(list(path.glob("*.safetensors")))


def run_command(cmd, dry_run=False):
    cmd = [str(item) for item in cmd]
    print(">>> " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--out_root", default="replication_runs/qwen25_7b_a100")
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--relative_layer", type=float, default=20 / 24)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n_random", type=int, default=8)
    ap.add_argument("--eval_max_blocks", type=int, default=128)
    ap.add_argument("--eval_batch", type=int, default=1)
    ap.add_argument("--extract_batch", type=int, default=1)
    ap.add_argument("--new_train_fraction", type=float, default=0.2)
    ap.add_argument("--probe_languages", nargs="+", default=["en", "zh", "fr", "de", "es"])
    ap.add_argument("--probe_train_per_lang", type=int, default=1200)
    ap.add_argument("--probe_test_per_lang", type=int, default=1200)
    ap.add_argument("--aligned_examples", type=int, default=2000)
    ap.add_argument("--vicreg_epochs", type=int, default=300)
    ap.add_argument("--optimizer", choices=["adamw8bit", "paged_adamw8bit"],
                    default="adamw8bit")
    ap.add_argument("--through", choices=STAGES, default="step7",
                    help="Run through stage; Step6 runs last to save compute.")
    ap.add_argument("--force", action="store_true",
                    help="Rerun subspace and intervention stages (not training).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands, without GPU or model downloads.")
    args = ap.parse_args()
    if not 0 < args.relative_layer <= 1:
        ap.error("--relative_layer must be in (0, 1].")
    if args.rank < 64:
        ap.error("--rank must be >=64 for two disjoint top/bottom rank-32 groups.")
    if args.n_random < 2:
        ap.error("--n_random must be >=2.")
    if not 0 < args.new_train_fraction <= 1:
        ap.error("--new_train_fraction must be in (0,1].")

    if args.dry_run:
        depth = 28  # Known Qwen2.5-7B layers. No network use.
    else:
        from transformers import AutoConfig
        depth = int(AutoConfig.from_pretrained(args.model_name).num_hidden_layers)
    layer = max(1, min(depth, round(depth * args.relative_layer)))
    run_dir = ROOT / args.out_root / f"seed{args.seed}"
    train_dir, sub_dir = run_dir / "training", run_dir / "subspaces"
    step7_dir, step6_dir = run_dir / "drift_isr_partition", run_dir / "energy_controls"
    anchor, adapted = train_dir / "anchor", train_dir / "adapted"
    subspaces = sub_dir / "core_subspaces.pt"

    commands = [
        ("training", [
            sys.executable, HERE / "train_7b_a100.py",
            "--model_name", args.model_name,
            "--data_dir", args.data_dir,
            "--seed", args.seed,
            "--old_language", args.old_language,
            "--new_language", args.new_language,
            "--new_train_fraction", args.new_train_fraction,
            "--eval_max_blocks", args.eval_max_blocks,
            "--eval_batch", args.eval_batch,
            "--optimizer", args.optimizer,
            "--out_dir", train_dir,
        ], train_dir / "training_summary.json"),
        ("subspaces", [
            sys.executable,
            ROOT / "experiments/retention_subspace_replication/build_core_subspaces.py",
            "--anchor_checkpoint", anchor,
            "--adapted_checkpoint", adapted,
            "--languages", *args.probe_languages,
            "--layer", layer,
            "--rank", args.rank,
            "--extract_batch", args.extract_batch,
            "--probe_train_per_lang", args.probe_train_per_lang,
            "--probe_test_per_lang", args.probe_test_per_lang,
            "--aligned_examples", args.aligned_examples,
            "--vicreg_epochs", args.vicreg_epochs,
            "--out_dir", sub_dir,
        ], subspaces),
        ("step7", [
            sys.executable,
            ROOT / "experiments/retention_subspace_mechanism/run_step7_drift_isr_partition_rescue.py",
            "--anchor_checkpoint", anchor,
            "--adapted_checkpoint", adapted,
            "--subspace_file", subspaces,
            "--data_dir", args.data_dir,
            "--old_language", args.old_language,
            "--new_language", args.new_language,
            "--ks", 16, 32,
            "--alphas", 0.25, 0.5, 1.0,
            "--n_random", args.n_random,
            "--random_seed", 9700 + 100 * args.seed,
            "--eval_max_blocks", args.eval_max_blocks,
            "--eval_batch", args.eval_batch,
            "--out_dir", step7_dir,
        ], step7_dir / "partition_rescue_summary.csv"),
        ("step6", [
            sys.executable,
            ROOT / "experiments/retention_subspace_mechanism/run_step6_energy_matched_controls.py",
            "--anchor_checkpoint", anchor,
            "--adapted_checkpoint", adapted,
            "--subspace_file", subspaces,
            "--data_dir", args.data_dir,
            "--old_language", args.old_language,
            "--new_language", args.new_language,
            "--subspaces", "transfer", "drift", "isr_cov", "isr_multiclass", "vicreg",
            "--strengths", 0.25, 0.5, 1.0,
            "--n_random", args.n_random,
            "--random_seed", 7300 + 100 * args.seed,
            "--eval_max_blocks", args.eval_max_blocks,
            "--eval_batch", args.eval_batch,
            "--out_dir", step6_dir,
        ], step6_dir / "energy_matched_summary.csv"),
    ]
    selected = STAGES[:STAGES.index(args.through) + 1]
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "run_manifest.json"
        manifest = vars(args).copy()
        manifest.update({
            "n_layers": depth, "selected_layer": layer,
            "effective_batch": 16,
            "note": "8-bit AdamW full-parameter FT; NOT exact standard AdamW.",
        })
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text())
            ignore = {"through", "dry_run", "force"}
            changed = [key for key, value in manifest.items()
                       if key not in ignore and previous.get(key) != value]
            if changed:
                raise ValueError("Output uses incompatible arguments: "
                                 f"{changed}. Choose a new --out_root.")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[setup] seed={args.seed} depth={depth} layer={layer} "
          f"stages={selected}", flush=True)
    for stage, cmd, output in commands:
        if stage not in selected:
            break
        done = output.is_file()
        if stage == "training":
            done = done and checkpoint_ok(anchor) and checkpoint_ok(adapted)
        if done and not args.force:
            print(f"[skip] already completed {stage}: {output}", flush=True)
            continue
        if args.force and stage == "training" and not args.dry_run:
            raise ValueError("Training --force intentionally disabled; "
                             "use a fresh --out_root instead.")
        if args.force and stage == "subspaces":
            cmd.append("--force")
        run_command(cmd, dry_run=args.dry_run)
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
