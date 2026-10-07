#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.random_layer_freeze_pilot.run_pilot import (  # noqa: E402
    load_model,
    seed_all,
    train,
)
from invariance.train_sequence import evaluate, load_blocks  # noqa: E402


def is_pareto(rows, idx):
    a = rows[idx]
    for j, b in enumerate(rows):
        if j == idx:
            continue
        weak_better = (
            b["forgetting"] <= a["forgetting"]
            and b["new_language_gain"] >= a["new_language_gain"]
        )
        strict_better = (
            b["forgetting"] < a["forgetting"]
            or b["new_language_gain"] > a["new_language_gain"]
        )
        if weak_better and strict_better:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Step 1: normalized layer/lambda sweep with no parameter freezing."
    )
    ap.add_argument(
        "--anchor_checkpoint",
        default="invariance_runs/sequence_seed0/en__zh/stage1_en",
    )
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 20, 24])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[5.0, 20.0, 50.0])
    ap.add_argument("--train_fraction", type=float, default=0.20)
    ap.add_argument("--eval_max_blocks", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_shuffle_seed", type=int, default=1001)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--normalization_eps", type=float, default=1e-8)
    ap.add_argument(
        "--out_dir",
        default="mechanism_runs/step1_layer_lambda_sweep",
    )
    ap.add_argument(
        "--save_all_checkpoints",
        action="store_true",
        help="By default only the canonical full_ft checkpoint is saved.",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if not 0 < args.train_fraction <= 1:
        raise ValueError("--train_fraction must be in (0,1].")
    if any(x <= 0 for x in args.lambdas):
        raise ValueError("All lambdas must be > 0.")

    # Attributes consumed by the shared train() helper.
    args.lambda_preserve = 0.0
    args.preservation_normalization = "anchor_energy"

    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_root = out / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    new_train = load_blocks(Path(args.data_dir) / f"{args.new_language}_train.pt")
    old_val = load_blocks(Path(args.data_dir) / f"{args.old_language}_val.pt")
    new_val = load_blocks(Path(args.data_dir) / f"{args.new_language}_val.pt")

    if args.eval_max_blocks > 0:
        old_val = old_val[: args.eval_max_blocks]
        new_val = new_val[: args.eval_max_blocks]

    g = torch.Generator().manual_seed(args.data_shuffle_seed)
    perm = torch.randperm(len(new_train), generator=g)
    n_train = max(1, round(len(new_train) * args.train_fraction))
    new_train = new_train[perm[:n_train]]

    tok = AutoTokenizer.from_pretrained(args.anchor_checkpoint, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    anchor = load_model(args.anchor_checkpoint, device, use_bf16)
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)

    anchor_old = evaluate(anchor, old_val, args.eval_batch, device, use_bf16)
    anchor_new = evaluate(anchor, new_val, args.eval_batch, device, use_bf16)
    print(
        f"[anchor] old={anchor_old:.6f} new={anchor_new:.6f} "
        f"train_blocks={len(new_train)} fraction={args.train_fraction:.3f}"
    )

    jobs = [("full_ft", None, 0.0)]
    for layer in args.layers:
        for lam in args.lambdas:
            jobs.append((f"layer{layer}_lam{lam:g}_norm", layer, float(lam)))

    rows = []
    for k, (name, layer, lam) in enumerate(jobs, 1):
        print(f"\n=== [{k}/{len(jobs)}] {name} ===")
        seed_all(args.seed)
        model = load_model(args.anchor_checkpoint, device, use_bf16)

        args.lambda_preserve = lam
        stats = train(
            model,
            anchor,
            new_train,
            args,
            device,
            use_bf16,
            mask=None,
            preserve_layer=layer,
        )

        post_old = evaluate(model, old_val, args.eval_batch, device, use_bf16)
        post_new = evaluate(model, new_val, args.eval_batch, device, use_bf16)
        forgetting = post_old - anchor_old
        gain = anchor_new - post_new

        print(
            f"forget={forgetting:+.6f} gain={gain:+.6f} "
            f"pres={stats['mean_preserve_loss']:.6f} "
            f"raw={stats['mean_raw_preserve_mse']:.6f} "
            f"anchor_E={stats['mean_anchor_energy']:.6f}"
        )

        should_save = name == "full_ft" or args.save_all_checkpoints
        if should_save:
            d = ckpt_root / name
            d.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(d)
            tok.save_pretrained(d)

        rows.append(
            {
                "condition": name,
                "target_layer_1based": "" if layer is None else layer,
                "lambda_preserve": lam,
                "preservation_normalization": (
                    "none" if layer is None else "anchor_energy"
                ),
                "train_fraction": args.train_fraction,
                "train_blocks": len(new_train),
                "tokens_seen": stats["tokens_seen"],
                "mean_train_lm_loss": stats["mean_train_lm_loss"],
                "mean_preserve_loss": stats["mean_preserve_loss"],
                "mean_raw_preserve_mse": stats["mean_raw_preserve_mse"],
                "mean_anchor_energy": stats["mean_anchor_energy"],
                "anchor_old_loss": anchor_old,
                "anchor_new_loss": anchor_new,
                "post_old_loss": post_old,
                "post_new_loss": post_new,
                "forgetting": forgetting,
                "new_language_gain": gain,
            }
        )

        del model
        torch.cuda.empty_cache()

        with (out / "results_partial.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    base = next(r for r in rows if r["condition"] == "full_ft")
    for r in rows:
        r["forgetting_reduction_vs_full_ft"] = base["forgetting"] - r["forgetting"]
        r["plasticity_cost_vs_full_ft"] = (
            base["new_language_gain"] - r["new_language_gain"]
        )

    for i, r in enumerate(rows):
        r["pareto_nondominated"] = is_pareto(rows, i)

    fields = list(rows[0].keys())
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    manifest = vars(args).copy()
    manifest["canonical_adapted_checkpoint"] = str(ckpt_root / "full_ft")
    manifest["scientific_question"] = (
        "Does the mid-layer retention/plasticity advantage persist across "
        "normalized preservation strengths without parameter freezing?"
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== Pareto summary ===")
    for r in sorted(rows, key=lambda x: (x["forgetting"], -x["new_language_gain"])):
        mark = "*" if r["pareto_nondominated"] else " "
        print(
            f"{mark} {r['condition']:24s} "
            f"forget={r['forgetting']:+.6f} "
            f"gain={r['new_language_gain']:+.6f}"
        )

    print(f"\nSaved: {out / 'results.csv'}")
    print(f"Canonical adapted checkpoint: {ckpt_root / 'full_ft'}")


if __name__ == "__main__":
    main()
