#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.random_layer_freeze_pilot.run_pilot import get_layers, load_model  # noqa: E402
from invariance.train_sequence import evaluate, load_blocks, make_loader  # noqa: E402


def replace_hidden(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden, *output[1:])
    return new_hidden


@torch.no_grad()
def evaluate_rescue(
    adapted,
    anchor,
    blocks,
    batch_size,
    device,
    use_bf16,
    *,
    layer_no,
    basis,
    alpha,
    scale=1.0,
):
    adapted.eval()
    anchor.eval()

    q = basis.to(device=device, dtype=torch.float32)
    anchor_layer = get_layers(anchor)[layer_no - 1]
    adapted_layer = get_layers(adapted)[layer_no - 1]

    shared = {"anchor_hidden": None}
    stats = {
        "projected_delta_sq": 0.0,
        "full_delta_sq": 0.0,
        "n": 0,
    }

    def anchor_hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        shared["anchor_hidden"] = h.detach()

    def adapted_hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if shared["anchor_hidden"] is None:
            raise RuntimeError("Anchor hidden state not available before adapted forward.")

        ha = shared["anchor_hidden"]
        if ha.shape != h.shape:
            raise RuntimeError(
                f"Anchor/adapted hidden shape mismatch: {ha.shape} vs {h.shape}"
            )

        delta = ha.float() - h.float()
        projected = (delta @ q) @ q.T
        perturb = alpha * scale * projected
        rescued = h.float() + perturb

        stats["projected_delta_sq"] += float(perturb.pow(2).sum())
        stats["full_delta_sq"] += float(delta.pow(2).sum())
        stats["n"] += delta.numel()

        return replace_hidden(output, rescued.to(dtype=h.dtype))

    h_anchor = anchor_layer.register_forward_hook(anchor_hook)
    h_adapted = adapted_layer.register_forward_hook(adapted_hook)

    total_loss = 0.0
    total_targets = 0
    try:
        for (x,) in make_loader(blocks, batch_size):
            x = x.to(device, non_blocking=True)
            shared["anchor_hidden"] = None

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda" and use_bf16),
            ):
                _ = anchor(input_ids=x, use_cache=False)
                out = adapted(input_ids=x, labels=x, use_cache=False)

            n = x.shape[0] * (x.shape[1] - 1)
            total_loss += float(out.loss) * n
            total_targets += n
    finally:
        h_anchor.remove()
        h_adapted.remove()

    fraction = stats["projected_delta_sq"] / max(stats["full_delta_sq"], 1e-30)
    return total_loss / total_targets, fraction


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Step 4: rescue adapted representations toward the anchor only "
            "inside selected Layer-20 subspaces."
        )
    )
    ap.add_argument(
        "--anchor_checkpoint",
        default="invariance_runs/sequence_seed0/en__zh/stage1_en",
    )
    ap.add_argument(
        "--adapted_checkpoint",
        default="mechanism_runs/step1_layer_lambda_sweep/checkpoints/full_ft",
    )
    ap.add_argument(
        "--subspace_file",
        default="mechanism_runs/step2_layer20_subspaces/layer20_subspaces.pt",
    )
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        help="Fraction of anchor-minus-adapted projected delta restored.",
    )
    ap.add_argument("--eval_max_blocks", type=int, default=128)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument(
        "--out_dir",
        default="mechanism_runs/step4_causal_rescue",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if any(a <= 0 or a > 1 for a in args.alphas):
        raise ValueError("--alphas must be in (0,1].")

    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.subspace_file, map_location="cpu")
    layer_no = int(payload["layer"])
    subspaces = {k: v.float() for k, v in payload["subspaces"].items()}
    real_subspaces = payload.get(
        "real_subspaces",
        [k for k in subspaces if not k.startswith("random_")],
    )
    matched_controls = payload.get(
        "matched_random_controls",
        {k: f"random_{k}" for k in real_subspaces},
    )

    langs = [args.old_language, args.new_language]
    val = {}
    for lang in langs:
        blocks = load_blocks(Path(args.data_dir) / f"{lang}_val.pt")
        if args.eval_max_blocks > 0:
            blocks = blocks[: args.eval_max_blocks]
        val[lang] = blocks

    print("Loading models...")
    anchor = load_model(args.anchor_checkpoint, device, use_bf16)
    adapted = load_model(args.adapted_checkpoint, device, use_bf16)
    anchor.eval()
    adapted.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    for p in adapted.parameters():
        p.requires_grad_(False)

    anchor_baseline = {}
    adapted_baseline = {}
    for lang in langs:
        anchor_baseline[lang] = evaluate(
            anchor, val[lang], args.eval_batch, device, use_bf16
        )
        adapted_baseline[lang] = evaluate(
            adapted, val[lang], args.eval_batch, device, use_bf16
        )
        print(
            f"[baseline] {lang} anchor={anchor_baseline[lang]:.6f} "
            f"adapted={adapted_baseline[lang]:.6f}"
        )

    rows = []
    for lang in langs:
        rows.append(
            {
                "subspace": "none",
                "matched_control": "",
                "rank": 0,
                "alpha": 0.0,
                "language": lang,
                "anchor_loss": anchor_baseline[lang],
                "adapted_baseline_loss": adapted_baseline[lang],
                "rescued_loss": adapted_baseline[lang],
                "loss_change_vs_adapted": 0.0,
                "recovery_toward_anchor": 0.0,
                "projected_delta_energy_fraction": 0.0,
            }
        )

    for name, q in subspaces.items():
        match = ""
        if name in matched_controls:
            match = matched_controls[name]
        elif name in matched_controls.values():
            match = next(
                real for real, ctrl in matched_controls.items()
                if ctrl == name
            )

        for alpha in args.alphas:
            for lang in langs:
                loss, projected_fraction = evaluate_rescue(
                    adapted,
                    anchor,
                    val[lang],
                    args.eval_batch,
                    device,
                    use_bf16,
                    layer_no=layer_no,
                    basis=q,
                    alpha=alpha,
                )

                # Positive recovery means the intervention moves loss toward the
                # anchor on this language. For the old language this is the
                # desired retention rescue. On the new language it may be a cost.
                denom = adapted_baseline[lang] - anchor_baseline[lang]
                if abs(denom) > 1e-12:
                    recovery = (
                        adapted_baseline[lang] - loss
                    ) / denom
                else:
                    recovery = float("nan")

                rows.append(
                    {
                        "subspace": name,
                        "matched_control": match,
                        "rank": int(q.shape[1]),
                        "alpha": alpha,
                        "language": lang,
                        "anchor_loss": anchor_baseline[lang],
                        "adapted_baseline_loss": adapted_baseline[lang],
                        "rescued_loss": loss,
                        "loss_change_vs_adapted": (
                            loss - adapted_baseline[lang]
                        ),
                        "recovery_toward_anchor": recovery,
                        "projected_delta_energy_fraction": projected_fraction,
                    }
                )
                print(
                    f"[rescue] {name:18s} rank={q.shape[1]:>3d} "
                    f"alpha={alpha:.2f} lang={lang} "
                    f"delta={loss-adapted_baseline[lang]:+.6f} "
                    f"recovery={recovery:+.4f} "
                    f"proj_frac={projected_fraction:.4f}"
                )

    results_file = out / "rescue_results.csv"
    with results_file.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Compare every real subspace against its matched random basis and compute
    # the old-language rescue / new-language plasticity tradeoff.
    paired = []
    for real in real_subspaces:
        random_name = matched_controls[real]
        for alpha in args.alphas:
            rr_old = next(
                r for r in rows
                if r["subspace"] == real
                and r["alpha"] == alpha
                and r["language"] == args.old_language
            )
            rc_old = next(
                r for r in rows
                if r["subspace"] == random_name
                and r["alpha"] == alpha
                and r["language"] == args.old_language
            )
            rr_new = next(
                r for r in rows
                if r["subspace"] == real
                and r["alpha"] == alpha
                and r["language"] == args.new_language
            )
            rc_new = next(
                r for r in rows
                if r["subspace"] == random_name
                and r["alpha"] == alpha
                and r["language"] == args.new_language
            )

            real_old_improvement = -rr_old["loss_change_vs_adapted"]
            random_old_improvement = -rc_old["loss_change_vs_adapted"]
            real_new_cost = rr_new["loss_change_vs_adapted"]
            random_new_cost = rc_new["loss_change_vs_adapted"]

            paired.append(
                {
                    "real_subspace": real,
                    "random_control": random_name,
                    "rank": rr_old["rank"],
                    "alpha": alpha,
                    "real_old_loss_improvement": real_old_improvement,
                    "random_old_loss_improvement": random_old_improvement,
                    "old_rescue_excess_vs_random": (
                        real_old_improvement - random_old_improvement
                    ),
                    "real_new_language_cost": real_new_cost,
                    "random_new_language_cost": random_new_cost,
                    "new_cost_excess_vs_random": (
                        real_new_cost - random_new_cost
                    ),
                    "real_old_recovery_fraction": rr_old["recovery_toward_anchor"],
                    "real_new_recovery_fraction": rr_new["recovery_toward_anchor"],
                    "real_projected_delta_fraction_old": rr_old[
                        "projected_delta_energy_fraction"
                    ],
                    "real_projected_delta_fraction_new": rr_new[
                        "projected_delta_energy_fraction"
                    ],
                }
            )

    paired_file = out / "matched_random_rescue_comparison.csv"
    with paired_file.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        w.writeheader()
        w.writerows(paired)

    meta = {
        "layer": layer_no,
        "anchor_checkpoint": args.anchor_checkpoint,
        "adapted_checkpoint": args.adapted_checkpoint,
        "subspace_file": args.subspace_file,
        "old_language": args.old_language,
        "new_language": args.new_language,
        "alphas": args.alphas,
        "real_subspaces": real_subspaces,
        "matched_random_controls": matched_controls,
        "intervention": (
            "h_adapt <- h_adapt + alpha * P_S(h_anchor - h_adapt) "
            "at the selected transformer block output"
        ),
        "interpretation": (
            "A useful retention-critical subspace should rescue old-language loss "
            "more than its matched random control while adding little new-language cost."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))

    print("\n=== Matched-random rescue at alpha=1.0 ===")
    for r in paired:
        if abs(float(r["alpha"]) - 1.0) < 1e-12:
            print(
                f"{r['real_subspace']:9s} "
                f"old_excess={r['old_rescue_excess_vs_random']:+.6f} "
                f"new_cost={r['real_new_language_cost']:+.6f} "
                f"old_recovery={r['real_old_recovery_fraction']:+.4f}"
            )
    print(f"\nSaved: {results_file}")
    print(f"Saved: {paired_file}")


if __name__ == "__main__":
    main()
