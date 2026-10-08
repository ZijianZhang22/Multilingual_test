#!/usr/bin/env python3
"""Run one complete paper-main replication for one Qwen size and one seed."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

EXACT_REFERENCE = {
    "anchor_old_loss": 2.499183,
    "anchor_new_loss": 2.900429,
    "adapted_old_loss": 2.507471,
    "adapted_new_loss": 2.824390,
    "forgetting": 0.008288,
    "new_language_gain": 0.076039,
}


def run(cmd, log_file):
    cmd = [str(x) for x in cmd]
    print("\n>>>", " ".join(cmd), flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n>>> " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
        rc = p.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def complete_checkpoint(path):
    p = Path(path)
    return (p / "config.json").exists() and any(p.glob("*.safetensors"))


def write_exact_reference_comparison(train_dir, run_dir):
    summary_file = train_dir / "training_summary.json"
    if not summary_file.exists():
        return
    got = json.loads(summary_file.read_text())
    comparison = {}
    deltas = []
    for key, ref in EXACT_REFERENCE.items():
        val = float(got[key])
        delta = val - ref
        deltas.append(abs(delta))
        comparison[key] = {
            "legacy_reference": ref,
            "reproduced": val,
            "difference": delta,
            "abs_difference": abs(delta),
        }
    payload = {
        "comparison": comparison,
        "max_abs_difference": max(deltas),
        "note": (
            "Reference values are the original Qwen2.5-0.5B seed-0 EN->ZH run. "
            "This check is descriptive and does not hard-fail because CUDA, "
            "PyTorch, and Transformers versions can introduce small numerical differences."
        ),
    }
    (run_dir / "exact_reference_comparison.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(
        f"[exact-check] max absolute baseline/training difference = "
        f"{payload['max_abs_difference']:.8f}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--out_root", default="replication_runs/qwen_scale_seed")
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--relative_layer", type=float, default=20 / 24)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n_random", type=int, default=8)
    ap.add_argument("--new_train_fraction", type=float, default=0.20)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--eval_batch", type=int, default=4)
    ap.add_argument("--extract_batch", type=int, default=4)
    ap.add_argument("--eval_max_blocks", type=int, default=128)
    ap.add_argument(
        "--exact_0p5_seed0",
        action="store_true",
        help=(
            "Strict protocol mode for the original Qwen2.5-0.5B seed-0 run: "
            "forces seed=0, layer20/24, micro_batch=4, grad_accum=4, "
            "eval_batch=8, extract_batch=16, rank64, n_random=8, and no "
            "gradient checkpointing."
        ),
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.exact_0p5_seed0:
        if args.model_name != "Qwen/Qwen2.5-0.5B":
            raise ValueError(
                "--exact_0p5_seed0 requires --model_name Qwen/Qwen2.5-0.5B"
            )
        if args.seed != 0:
            raise ValueError("--exact_0p5_seed0 requires --seed 0")
        if args.old_language != "en" or args.new_language != "zh":
            raise ValueError("--exact_0p5_seed0 is defined for EN->ZH only.")
        args.relative_layer = 20 / 24
        args.rank = 64
        args.n_random = 8
        args.new_train_fraction = 0.20
        args.lr = 2e-5
        args.weight_decay = 0.1
        args.micro_batch = 4
        args.grad_accum = 4
        args.eval_batch = 8
        args.extract_batch = 16
        args.eval_max_blocks = 128
        print("[mode] exact Qwen2.5-0.5B seed-0 reproduction", flush=True)

    cfg = AutoConfig.from_pretrained(args.model_name)
    n_layers = int(getattr(cfg, "num_hidden_layers"))
    layer = max(1, min(n_layers, round(n_layers * args.relative_layer)))

    if args.exact_0p5_seed0 and layer != 20:
        raise RuntimeError(
            f"Exact mode expected Layer 20 but model config selected Layer {layer}."
        )

    run_dir = Path(args.out_root) / args.model_tag / f"seed{args.seed}"
    train_dir = run_dir / "training"
    sub_dir = run_dir / "subspaces"
    energy_dir = run_dir / "energy_controls"
    step7_dir = run_dir / "drift_isr_partition"
    log = run_dir / "replication.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 0.5B exact/seed replications keep the original no-GC training path.
    # Larger models use GC only for memory; effective batch size remains 16.
    use_gradient_checkpointing = args.model_name != "Qwen/Qwen2.5-0.5B"

    meta = {
        "model_name": args.model_name,
        "model_tag": args.model_tag,
        "seed": args.seed,
        "exact_0p5_seed0": args.exact_0p5_seed0,
        "n_layers": n_layers,
        "relative_layer_target": args.relative_layer,
        "selected_layer_1based": layer,
        "selected_relative_depth": layer / n_layers,
        "rank": args.rank,
        "n_random": args.n_random,
        "old_language": args.old_language,
        "new_language": args.new_language,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "effective_batch": args.micro_batch * args.grad_accum,
        "eval_batch": args.eval_batch,
        "extract_batch": args.extract_batch,
        "gradient_checkpointing": use_gradient_checkpointing,
        "protocol": (
            "exact_0p5_seed0"
            if args.exact_0p5_seed0
            else "controlled_scale_or_seed_replication"
        ),
    }
    (run_dir / "replication_manifest.json").write_text(json.dumps(meta, indent=2))

    print(
        f"[replication] {args.model_tag} seed={args.seed}: "
        f"{n_layers} layers -> layer {layer} ({layer/n_layers:.3f}); "
        f"effective_batch={args.micro_batch * args.grad_accum}",
        flush=True,
    )

    anchor = train_dir / "anchor"
    adapted = train_dir / "adapted"

    if args.force or not (
        complete_checkpoint(anchor) and complete_checkpoint(adapted)
    ):
        cmd = [
            sys.executable,
            HERE / "train_anchor_adapt.py",
            "--model_name",
            args.model_name,
            "--data_dir",
            args.data_dir,
            "--old_language",
            args.old_language,
            "--new_language",
            args.new_language,
            "--seed",
            args.seed,
            "--new_train_fraction",
            args.new_train_fraction,
            "--lr",
            args.lr,
            "--weight_decay",
            args.weight_decay,
            "--micro_batch",
            args.micro_batch,
            "--grad_accum",
            args.grad_accum,
            "--eval_batch",
            args.eval_batch,
            "--eval_max_blocks",
            args.eval_max_blocks,
            "--reload_anchor_before_new_stage",
            "--out_dir",
            train_dir,
        ]
        if use_gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        run(cmd, log)
    else:
        print("[resume] training checkpoints already exist", flush=True)

    if args.exact_0p5_seed0:
        write_exact_reference_comparison(train_dir, run_dir)

    subspace_file = sub_dir / "core_subspaces.pt"
    if args.force or not subspace_file.exists():
        cmd = [
            sys.executable,
            HERE / "build_core_subspaces.py",
            "--anchor_checkpoint",
            anchor,
            "--adapted_checkpoint",
            adapted,
            "--layer",
            layer,
            "--rank",
            args.rank,
            "--extract_batch",
            args.extract_batch,
            "--out_dir",
            sub_dir,
        ]
        if args.force:
            cmd.append("--force")
        run(cmd, log)
    else:
        print("[resume] core subspaces already exist", flush=True)

    # Keep the original Step-6 subspace order because the random-control seed
    # offset depends on the subspace index.
    energy_summary = energy_dir / "energy_matched_summary.csv"
    if args.force or not energy_summary.exists():
        cmd = [
            sys.executable,
            ROOT
            / "experiments/retention_subspace_mechanism/run_step6_energy_matched_controls.py",
            "--anchor_checkpoint",
            anchor,
            "--adapted_checkpoint",
            adapted,
            "--subspace_file",
            subspace_file,
            "--data_dir",
            args.data_dir,
            "--old_language",
            args.old_language,
            "--new_language",
            args.new_language,
            "--subspaces",
            "transfer",
            "drift",
            "isr_cov",
            "isr_multiclass",
            "vicreg",
            "--strengths",
            "0.25",
            "0.5",
            "1.0",
            "--n_random",
            args.n_random,
            "--random_seed",
            7300 + 100 * args.seed,
            "--eval_max_blocks",
            args.eval_max_blocks,
            "--eval_batch",
            args.eval_batch,
            "--out_dir",
            energy_dir,
        ]
        run(cmd, log)
    else:
        print("[resume] energy-matched Step-6 controls already exist", flush=True)

    step7_summary = step7_dir / "partition_rescue_summary.csv"
    if args.force or not step7_summary.exists():
        cmd = [
            sys.executable,
            ROOT
            / "experiments/retention_subspace_mechanism/run_step7_drift_isr_partition_rescue.py",
            "--anchor_checkpoint",
            anchor,
            "--adapted_checkpoint",
            adapted,
            "--subspace_file",
            subspace_file,
            "--data_dir",
            args.data_dir,
            "--old_language",
            args.old_language,
            "--new_language",
            args.new_language,
            "--ks",
            "16",
            "32",
            "--alphas",
            "0.25",
            "0.5",
            "1.0",
            "--n_random",
            args.n_random,
            "--random_seed",
            9700 + 100 * args.seed,
            "--eval_max_blocks",
            args.eval_max_blocks,
            "--eval_batch",
            args.eval_batch,
            "--out_dir",
            step7_dir,
        ]
        run(cmd, log)
    else:
        print("[resume] Step-7 Drift/ISR partition already exists", flush=True)

    print(f"\nDONE: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
