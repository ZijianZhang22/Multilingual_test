#!/usr/bin/env python3
"""Step 7: decompose Drift by alignment with ISR-Multiclass and test causal rescue.

Scientific question
-------------------
Does forgetting live preferentially in the component of adaptation-induced Drift
that is most aligned with the shared semantic ISR-Multiclass subspace?

We avoid calling the exact set-theoretic intersection of two finite-dimensional
estimated subspaces an "intersection" because it is typically numerically empty.
Instead we diagonalize the Drift->ISR projection operator

    A = Q_drift^T P_ISR Q_drift

inside the Drift subspace. Its eigenvectors are Drift directions ordered by
squared cosine alignment with ISR-Multiclass.

For each k:
  - top-k:    the k Drift directions most aligned with ISR-Multiclass
  - bottom-k: the k Drift directions least aligned with ISR-Multiclass

Top-k and bottom-k have identical rank. We then rescue anchor-directed
adaptation displacement in those directions.

Two comparisons are reported:
  1) natural rescue (scale=1): how much the actually observed component matters;
  2) cross-component energy-matched rescue: top-k and bottom-k are scaled DOWN
     to the same projected-delta energy, then compared with rank-matched random
     directions calibrated to that same energy.

The second comparison is the primary specificity test.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.random_layer_freeze_pilot.run_pilot import load_model
from experiments.retention_subspace_mechanism.run_step4_causal_rescue import evaluate_rescue
from experiments.retention_subspace_mechanism.subspace_extractors import (
    orthonormal_random,
    orthonormalize,
)
from invariance.train_sequence import evaluate, load_blocks


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def drift_ordered_by_isr_alignment(q_drift, q_isr):
    """Return orthonormal Drift directions ordered by ISR alignment.

    For unit Drift direction q = Q_drift u,
        ||P_ISR q||^2 = u^T (Qd^T Qi Qi^T Qd) u.
    Thus eigenvectors of that operator give the principal Drift directions and
    eigenvalues are squared cosines to ISR.
    """
    qd = orthonormalize(q_drift)
    qi = orthonormalize(q_isr)
    a = qd.T @ qi @ qi.T @ qd
    a = 0.5 * (a + a.T)
    evals, evecs = torch.linalg.eigh(a)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp(min=0.0, max=1.0)
    evecs = evecs[:, order]
    q_ordered = orthonormalize(qd @ evecs)
    # QR can flip signs but preserves column order/span. Recompute alignment
    # scores directly for robust reporting.
    cos2 = (qi.T @ q_ordered).pow(2).sum(dim=0).clamp(0.0, 1.0)
    # Numerical QR should preserve ordering, but explicitly sort once more.
    order2 = torch.argsort(cos2, descending=True)
    return q_ordered[:, order2], cos2[order2]


def main():
    ap = argparse.ArgumentParser(
        description="Step 7: principal-angle Drift/ISR-Multiclass partition rescue."
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
        default="mechanism_runs/step2_layer20_subspaces_v2/layer20_subspaces.pt",
    )
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--n_random", type=int, default=8)
    ap.add_argument("--random_seed", type=int, default=9700)
    ap.add_argument("--eval_max_blocks", type=int, default=128)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--max_random_scale", type=float, default=8.0)
    ap.add_argument(
        "--out_dir",
        default="mechanism_runs/step7_drift_isr_partition",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.n_random < 2:
        raise ValueError("--n_random must be >= 2")
    if any(a <= 0 or a > 1 for a in args.alphas):
        raise ValueError("--alphas must be in (0, 1].")

    payload = torch.load(args.subspace_file, map_location="cpu")
    layer = int(payload["layer"])
    dim = int(payload["hidden_dim"])
    subspaces = payload["subspaces"]
    if "drift" not in subspaces or "isr_multiclass" not in subspaces:
        raise KeyError("subspace_file must contain 'drift' and 'isr_multiclass'.")

    q_drift = subspaces["drift"].float()
    q_isr = subspaces["isr_multiclass"].float()
    q_ordered, cos2 = drift_ordered_by_isr_alignment(q_drift, q_isr)
    drift_rank = q_ordered.shape[1]

    ks = sorted(set(args.ks))
    for k in ks:
        if k <= 0:
            raise ValueError("All k must be > 0.")
        if 2 * k > drift_rank:
            raise ValueError(
                f"k={k} invalid for Drift rank={drift_rank}: need 2*k <= rank "
                "so top-k and bottom-k are disjoint and rank matched."
            )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save geometry before any causal evaluation.
    angle_rows = []
    for i, c2 in enumerate(cos2.tolist(), 1):
        angle_rows.append(
            {
                "drift_principal_index": i,
                "cos2_with_isr_multiclass": c2,
                "cos_with_isr_multiclass": math.sqrt(max(c2, 0.0)),
                "principal_angle_deg": math.degrees(
                    math.acos(min(1.0, max(0.0, math.sqrt(max(c2, 0.0)))))
                ),
            }
        )
    write_csv(out / "principal_alignment_spectrum.csv", angle_rows)

    derived = {}
    geometry = {}
    for k in ks:
        q_top = q_ordered[:, :k].contiguous()
        q_bottom = q_ordered[:, -k:].contiguous()
        derived[f"drift_isr_top{k}"] = q_top
        derived[f"drift_isr_bottom{k}"] = q_bottom
        geometry[str(k)] = {
            "top_mean_cos2": float(cos2[:k].mean()),
            "bottom_mean_cos2": float(cos2[-k:].mean()),
            "top_min_cos2": float(cos2[:k].min()),
            "bottom_max_cos2": float(cos2[-k:].max()),
            "top_bottom_orthogonality_max_abs": float(
                (q_top.T @ q_bottom).abs().max()
            ),
        }

    torch.save(
        {
            "layer": layer,
            "hidden_dim": dim,
            "source_subspace_file": args.subspace_file,
            "drift_rank": drift_rank,
            "isr_rank": int(q_isr.shape[1]),
            "principal_cos2": cos2,
            "derived_subspaces": derived,
            "geometry": geometry,
            "definition": (
                "Drift directions are eigenvectors of Qd^T P_ISR Qd, ordered "
                "by squared projection onto ISR-Multiclass."
            ),
        },
        out / "drift_isr_partition.pt",
    )
    (out / "geometry.json").write_text(json.dumps(geometry, indent=2))

    print(
        f"[geometry] layer={layer} drift_rank={drift_rank} "
        f"mean_cos2={float(cos2.mean()):.6f}",
        flush=True,
    )
    for k in ks:
        g = geometry[str(k)]
        print(
            f"[geometry] k={k} top_mean_cos2={g['top_mean_cos2']:.6f} "
            f"bottom_mean_cos2={g['bottom_mean_cos2']:.6f}",
            flush=True,
        )

    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()

    val = {}
    for lang in [args.old_language, args.new_language]:
        blocks = load_blocks(Path(args.data_dir) / f"{lang}_val.pt")
        if args.eval_max_blocks > 0:
            blocks = blocks[: args.eval_max_blocks]
        val[lang] = blocks

    print("Loading anchor/adapted models...", flush=True)
    anchor = load_model(args.anchor_checkpoint, device, use_bf16)
    adapted = load_model(args.adapted_checkpoint, device, use_bf16)
    anchor.eval()
    adapted.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    for p in adapted.parameters():
        p.requires_grad_(False)

    base_anchor = {}
    base_adapted = {}
    for lang in val:
        base_anchor[lang] = evaluate(
            anchor, val[lang], args.eval_batch, device, use_bf16
        )
        base_adapted[lang] = evaluate(
            adapted, val[lang], args.eval_batch, device, use_bf16
        )
        print(
            f"[baseline] {lang} anchor={base_anchor[lang]:.6f} "
            f"adapted={base_adapted[lang]:.6f}",
            flush=True,
        )

    forgetting_gap = (
        base_adapted[args.old_language] - base_anchor[args.old_language]
    )
    new_gain = (
        base_anchor[args.new_language] - base_adapted[args.new_language]
    )

    summary_rows = []
    random_rows = []

    for k in ks:
        top_name = f"drift_isr_top{k}"
        bottom_name = f"drift_isr_bottom{k}"
        q_top = derived[top_name]
        q_bottom = derived[bottom_name]

        randoms = [
            (
                f"random_rank{k}_{j:02d}",
                orthonormal_random(dim, k, args.random_seed + 1000 * k + j),
            )
            for j in range(args.n_random)
        ]

        for lang in val:
            # Natural projected-delta energy of the two real components.
            _, top_frac = evaluate_rescue(
                adapted, anchor, val[lang], args.eval_batch, device, use_bf16,
                layer_no=layer, basis=q_top, alpha=1.0, scale=1.0,
            )
            _, bottom_frac = evaluate_rescue(
                adapted, anchor, val[lang], args.eval_batch, device, use_bf16,
                layer_no=layer, basis=q_bottom, alpha=1.0, scale=1.0,
            )

            # Primary controlled comparison: scale DOWN the larger real
            # component to the smaller component's energy. No real component
            # is ever amplified.
            target_frac = min(top_frac, bottom_frac)
            top_scale = math.sqrt(target_frac / max(top_frac, 1e-30))
            bottom_scale = math.sqrt(target_frac / max(bottom_frac, 1e-30))

            random_cal = []
            for rn, rq in randoms:
                _, rf = evaluate_rescue(
                    adapted, anchor, val[lang], args.eval_batch, device, use_bf16,
                    layer_no=layer, basis=rq, alpha=1.0, scale=1.0,
                )
                rscale = min(
                    math.sqrt(target_frac / max(rf, 1e-30)),
                    args.max_random_scale,
                )
                random_cal.append((rn, rq, rf, rscale))

            print(
                f"[calibrate] k={k} lang={lang} "
                f"top_frac={top_frac:.6f} bottom_frac={bottom_frac:.6f} "
                f"target={target_frac:.6f} "
                f"scales(top,bottom)=({top_scale:.3f},{bottom_scale:.3f})",
                flush=True,
            )

            for alpha in args.alphas:
                # Natural effects: actual contribution with no cross-component
                # energy normalization.
                top_nat_loss, _ = evaluate_rescue(
                    adapted, anchor, val[lang], args.eval_batch, device, use_bf16,
                    layer_no=layer, basis=q_top, alpha=alpha, scale=1.0,
                )
                bottom_nat_loss, _ = evaluate_rescue(
                    adapted, anchor, val[lang], args.eval_batch, device, use_bf16,
                    layer_no=layer, basis=q_bottom, alpha=alpha, scale=1.0,
                )
                top_nat_change = top_nat_loss - base_adapted[lang]
                bottom_nat_change = bottom_nat_loss - base_adapted[lang]

                # Primary: same rank + same projected-delta energy.
                top_loss, top_matched_frac = evaluate_rescue(
                    adapted, anchor, val[lang], args.eval_batch, device, use_bf16,
                    layer_no=layer, basis=q_top, alpha=alpha, scale=top_scale,
                )
                bottom_loss, bottom_matched_frac = evaluate_rescue(
                    adapted, anchor, val[lang], args.eval_batch, device, use_bf16,
                    layer_no=layer, basis=q_bottom, alpha=alpha, scale=bottom_scale,
                )
                top_change = top_loss - base_adapted[lang]
                bottom_change = bottom_loss - base_adapted[lang]

                rand_changes = []
                for rn, rq, raw_frac, rscale in random_cal:
                    rloss, rfrac = evaluate_rescue(
                        adapted, anchor, val[lang], args.eval_batch, device, use_bf16,
                        layer_no=layer, basis=rq, alpha=alpha, scale=rscale,
                    )
                    rchange = rloss - base_adapted[lang]
                    rand_changes.append(rchange)
                    random_rows.append(
                        {
                            "k": k,
                            "language": lang,
                            "alpha": alpha,
                            "control": rn,
                            "raw_random_fraction": raw_frac,
                            "random_scale": rscale,
                            "matched_random_fraction": rfrac,
                            "random_loss_change": rchange,
                        }
                    )

                rand_mean = mean(rand_changes)
                rand_std = std(rand_changes)

                row = {
                    "k": k,
                    "language": lang,
                    "alpha": alpha,
                    "top_mean_cos2": geometry[str(k)]["top_mean_cos2"],
                    "bottom_mean_cos2": geometry[str(k)]["bottom_mean_cos2"],
                    "top_natural_fraction_alpha1": top_frac,
                    "bottom_natural_fraction_alpha1": bottom_frac,
                    "matched_target_fraction_alpha1": target_frac,
                    "top_scale": top_scale,
                    "bottom_scale": bottom_scale,
                    "top_natural_loss_change": top_nat_change,
                    "bottom_natural_loss_change": bottom_nat_change,
                    "top_matched_loss_change": top_change,
                    "bottom_matched_loss_change": bottom_change,
                    "random_matched_mean_loss_change": rand_mean,
                    "random_matched_std_loss_change": rand_std,
                    # Positive means top-aligned directions rescue MORE than
                    # bottom-aligned directions at equal rank and energy.
                    "top_improvement_excess_vs_bottom": bottom_change - top_change,
                    "top_improvement_excess_vs_random": rand_mean - top_change,
                    "bottom_improvement_excess_vs_random": rand_mean - bottom_change,
                    "top_matched_projected_fraction": top_matched_frac,
                    "bottom_matched_projected_fraction": bottom_matched_frac,
                }

                if lang == args.old_language and forgetting_gap > 0:
                    row["top_old_recovery_fraction"] = -top_change / forgetting_gap
                    row["bottom_old_recovery_fraction"] = -bottom_change / forgetting_gap
                    row["random_old_recovery_fraction"] = -rand_mean / forgetting_gap
                else:
                    row["top_old_recovery_fraction"] = ""
                    row["bottom_old_recovery_fraction"] = ""
                    row["random_old_recovery_fraction"] = ""

                if lang == args.new_language and new_gain > 0:
                    row["top_new_gain_cost_fraction"] = top_change / new_gain
                    row["bottom_new_gain_cost_fraction"] = bottom_change / new_gain
                    row["random_new_gain_cost_fraction"] = rand_mean / new_gain
                else:
                    row["top_new_gain_cost_fraction"] = ""
                    row["bottom_new_gain_cost_fraction"] = ""
                    row["random_new_gain_cost_fraction"] = ""

                summary_rows.append(row)
                print(
                    f"[matched rescue] k={k:2d} {lang} alpha={alpha:.2f} "
                    f"top={top_change:+.6f} bottom={bottom_change:+.6f} "
                    f"random={rand_mean:+.6f}±{rand_std:.6f} "
                    f"top-vs-bottom={bottom_change-top_change:+.6f}",
                    flush=True,
                )

    write_csv(out / "partition_rescue_summary.csv", summary_rows)
    write_csv(out / "partition_random_draws.csv", random_rows)

    manifest = {
        "anchor_checkpoint": args.anchor_checkpoint,
        "adapted_checkpoint": args.adapted_checkpoint,
        "source_subspace_file": args.subspace_file,
        "layer": layer,
        "old_language": args.old_language,
        "new_language": args.new_language,
        "ks": ks,
        "alphas": args.alphas,
        "n_random": args.n_random,
        "random_seed": args.random_seed,
        "forgetting_gap": forgetting_gap,
        "new_language_gain": new_gain,
        "primary_test": (
            "At identical rank and projected rescue energy, do the Drift "
            "directions most aligned with ISR-Multiclass rescue old-language "
            "forgetting more than the least aligned Drift directions?"
        ),
        "energy_matching": (
            "For each k/language, top-k and bottom-k are both scaled down to "
            "min(top natural projected-delta energy, bottom natural projected-"
            "delta energy). Random rank-k bases are scaled to the same target."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== Primary old-language comparison ===", flush=True)
    for r in summary_rows:
        if r["language"] == args.old_language:
            print(
                f"k={r['k']:2d} alpha={r['alpha']:.2f} "
                f"top_rec={r['top_old_recovery_fraction']:+.3f} "
                f"bottom_rec={r['bottom_old_recovery_fraction']:+.3f} "
                f"random_rec={r['random_old_recovery_fraction']:+.3f} "
                f"top_excess_vs_bottom={r['top_improvement_excess_vs_bottom']:+.6f}",
                flush=True,
            )

    print(f"\nSaved: {out / 'partition_rescue_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
