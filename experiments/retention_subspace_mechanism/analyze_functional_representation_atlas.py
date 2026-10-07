#!/usr/bin/env python3
"""Layer-wise functional representation atlas for multilingual adaptation.

This analysis is intentionally separate from the causal Step-2/3/4 pipeline.
It asks:
  1) how much each layer changes globally,
  2) where that change lands in functional subspaces,
  3) how stable/rotated each functional subspace is,
  4) how much language/task information each subspace carries,
  5) how candidate subspaces overlap,
  6) at the causal layer, whether geometry agrees with removal/rescue effects.

The atlas uses a lighter per-layer subspace family than the detailed Layer-20
causal suite so it can be run across many layers:
  - language_linear: row-space of a linear language classifier
  - task_linear: row-space of a linear XNLI classifier
  - transfer: aligned semantic PCA after removing language_linear
  - drift: PCA of adapted-minus-anchor feature deltas
  - isr_cov: class-conditional covariance-invariant directions
  - isr_multiclass: multiclass/environment-derived invariant semantic directions
  - vicreg: VICReg-inspired linear cross-lingual invariant directions
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
INV = ROOT / "invariance"

from experiments.retention_subspace_mechanism.subspace_extractors import (  # noqa: E402
    fit_isr_cov,
    fit_isr_multiclass_semantic,
    fit_vicreg_linear,
    orthonormalize,
)
from invariance.benchmark_representation_methods import (  # noqa: E402
    train_linear_classifier,
)


def run(cmd):
    cmd = [str(x) for x in cmd]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def maybe(path, cmd, force=False):
    path = Path(path)
    if path.exists() and not force:
        print(f"SKIP existing: {path}")
        return
    run(cmd)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def centered(x):
    return x - x.mean(dim=0, keepdim=True)


def linear_cka(x, y):
    """Linear CKA using centered feature matrices without forming n x n Gram matrices."""
    x = centered(x.float())
    y = centered(y.float())
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    num = xy.pow(2).sum()
    den = torch.sqrt(xx.pow(2).sum() * yy.pow(2).sum()).clamp_min(1e-20)
    return float((num / den).cpu())


def relative_l2_drift(x_anchor, x_adapted):
    xa = centered(x_anchor.float())
    xb = centered(x_adapted.float())
    return float((xb - xa).norm() / xa.norm().clamp_min(1e-20))


def mean_example_cosine(x_anchor, x_adapted):
    xa = x_anchor.float()
    xb = x_adapted.float()
    return float(F.cosine_similarity(xa, xb, dim=1).mean())


def linear_alignment_discrepancy(x_ref, x_cur, ridge=1e-4, max_examples=2000):
    """Minimum normalized linear alignment error.

    Computes min_A ||X_ref - X_cur A||_F^2 / ||X_ref||_F^2 on centered,
    paired representations. This follows the linear-alignment idea behind
    representation discrepancy while exposing our exact normalization here.

    A symmetric version averages cur->ref and ref->cur.
    """
    x = centered(x_ref.float())
    y = centered(x_cur.float())

    if max_examples and x.shape[0] > max_examples:
        # Deterministic evenly spaced subsample.
        idx = torch.linspace(0, x.shape[0] - 1, max_examples).long()
        x = x[idx]
        y = y[idx]

    # Solve ridge least squares via d x d Gram matrices.
    # Move to GPU when available because hidden dim is typically 896.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)

    def one_way(target, source):
        d = source.shape[1]
        gram = source.T @ source
        scale = torch.trace(gram) / max(d, 1)
        reg = ridge * scale.clamp_min(1e-12)
        rhs = source.T @ target
        a = torch.linalg.solve(
            gram + reg * torch.eye(d, device=device, dtype=source.dtype),
            rhs,
        )
        residual = target - source @ a
        return residual.pow(2).sum() / target.pow(2).sum().clamp_min(1e-20)

    d_y_to_x = one_way(x, y)
    d_x_to_y = one_way(y, x)
    return (
        float(d_y_to_x.detach().cpu()),
        float(d_x_to_y.detach().cpu()),
        float(((d_y_to_x + d_x_to_y) / 2).detach().cpu()),
    )


def fit_classifier_basis(x, labels, train_idx, n_classes, *, epochs, seed):
    """Return the centered classifier row-space as a functional subspace."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.float().to(device)
    y = labels.long().to(device)
    train_idx = train_idx.to(device)
    head = train_linear_classifier(
        x[train_idx],
        y[train_idx],
        n_classes,
        epochs,
        1e-2,
        1e-4,
        seed,
    )
    w = head.weight.detach()
    w = w - w.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(w, full_matrices=False)
    rank = int((s > 1e-6).sum().item())
    if rank == 0:
        return torch.empty(x.shape[1], 0), {"rank": 0}
    q = vh[:rank].T.detach().cpu()
    q = orthonormalize(q)
    return q, {
        "rank": int(q.shape[1]),
        "singular_values": [float(v) for v in s[:rank].detach().cpu()],
    }


def fit_transfer_basis(x, rows, q_lang, rank):
    pair_to_indices = {}
    for i, row in enumerate(rows):
        pair_to_indices.setdefault(row["pair_id"], []).append(i)

    semantic_means = []
    for idxs in pair_to_indices.values():
        if len(idxs) < 2:
            continue
        semantic_means.append(x[idxs].mean(dim=0))
    m = torch.stack(semantic_means).float()

    if q_lang.numel():
        m = m - (m @ q_lang) @ q_lang.T
    m = centered(m)

    q = min(rank, m.shape[0], m.shape[1])
    _, s, v = torch.pca_lowrank(m, q=q, center=False)
    basis = v[:, :q]
    if q_lang.numel():
        basis = basis - q_lang @ (q_lang.T @ basis)
    basis = orthonormalize(basis, q)
    return basis, {
        "rank": int(basis.shape[1]),
        "n_semantic_pairs": int(m.shape[0]),
        "singular_values": [float(vv) for vv in s[: min(20, len(s))]],
    }


def fit_drift_basis(x_anchor, x_adapted, rank):
    delta = x_adapted.float() - x_anchor.float()
    dc = centered(delta)
    q = min(rank, dc.shape[0], dc.shape[1])
    _, s, v = torch.pca_lowrank(dc, q=q, center=False)
    basis = orthonormalize(v[:, :q], q)
    total_var = dc.pow(2).sum().clamp_min(1e-20)
    captured = (dc @ basis).pow(2).sum() / total_var
    return basis, {
        "rank": int(basis.shape[1]),
        "captured_drift_fraction": float(captured),
        "singular_values": [float(vv) for vv in s[: min(20, len(s))]],
    }


def principal_stats(qa, qb):
    if qa.numel() == 0 or qb.numel() == 0:
        return {
            "projection_overlap": float("nan"),
            "mean_cos2": float("nan"),
            "mean_angle_deg": float("nan"),
            "max_angle_deg": float("nan"),
            "min_rank": 0,
        }
    s = torch.linalg.svdvals(qa.float().T @ qb.float()).clamp(0, 1)
    r = min(qa.shape[1], qb.shape[1])
    overlap = float((s.pow(2).sum() / max(r, 1)).cpu())
    angles = torch.rad2deg(torch.acos(s))
    return {
        "projection_overlap": overlap,
        "mean_cos2": float(s.pow(2).mean().cpu()),
        "mean_angle_deg": float(angles.mean().cpu()),
        "max_angle_deg": float(angles.max().cpu()),
        "min_rank": int(r),
    }


def projected_drift_metrics(delta, q):
    if q.numel() == 0:
        return float("nan"), float("nan")
    d = delta.shape[1]
    r = q.shape[1]
    total = delta.pow(2).sum().clamp_min(1e-20)
    proj = (delta @ q).pow(2).sum()
    frac = proj / total
    # >1 means drift is enriched in this subspace relative to isotropic energy.
    enrichment = (frac / max(r / d, 1e-20))
    return float(frac), float(enrichment)


def probe_accuracy_in_subspace(
    x,
    q,
    labels,
    train_idx,
    test_idx,
    n_classes,
    *,
    epochs,
    seed,
):
    if q.numel() == 0:
        return float("nan")
    z = x.float() @ q.float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = z.to(device)
    y = labels.long().to(device)
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)
    head = train_linear_classifier(
        z[train_idx],
        y[train_idx],
        n_classes,
        epochs,
        1e-2,
        1e-4,
        seed,
    )
    with torch.no_grad():
        pred = head(z[test_idx]).argmax(dim=-1)
        acc = (pred == y[test_idx]).float().mean()
    return float(acc.detach().cpu())


def build_subspaces(
    x_probe,
    x_aligned,
    probe_payload,
    aligned_rows,
    *,
    rank,
    probe_epochs,
    isr_cov_class,
    vicreg_epochs,
    seed_offset,
):
    languages = probe_payload["languages"]
    unique_langs = sorted(set(languages))
    lang_to_id = {lang: i for i, lang in enumerate(unique_langs)}
    lang_ids = torch.tensor([lang_to_id[x] for x in languages], dtype=torch.long)
    labels = probe_payload["labels"].long()
    train_idx = torch.tensor(
        [i for i, s in enumerate(probe_payload["splits"]) if s == "probe_train"],
        dtype=torch.long,
    )

    q_lang, lang_meta = fit_classifier_basis(
        x_probe,
        lang_ids,
        train_idx,
        len(unique_langs),
        epochs=probe_epochs,
        seed=seed_offset + 1,
    )
    q_task, task_meta = fit_classifier_basis(
        x_probe,
        labels,
        train_idx,
        int(labels.max().item()) + 1,
        epochs=probe_epochs,
        seed=seed_offset + 2,
    )

    q_transfer, transfer_meta = fit_transfer_basis(
        x_aligned,
        aligned_rows,
        q_lang,
        rank,
    )

    train_x = x_probe[train_idx]
    train_langs = [languages[i] for i in train_idx.tolist()]
    train_labels = labels[train_idx]
    q_isr_cov, isr_cov_meta = fit_isr_cov(
        train_x,
        train_langs,
        train_labels,
        rank=rank,
        class_label=isr_cov_class,
    )
    q_isr_multi, _, isr_multi_meta = fit_isr_multiclass_semantic(
        train_x,
        train_langs,
        train_labels,
        rank=rank,
    )

    pair_ids = [r["pair_id"] for r in aligned_rows]
    q_vicreg, vicreg_meta = fit_vicreg_linear(
        x_aligned,
        pair_ids,
        rank=rank,
        epochs=vicreg_epochs,
        seed=seed_offset + 3,
    )

    return {
        "language_linear": q_lang,
        "task_linear": q_task,
        "transfer": q_transfer,
        "isr_cov": q_isr_cov,
        "isr_multiclass": q_isr_multi,
        "vicreg": q_vicreg,
    }, {
        "language_linear": lang_meta,
        "task_linear": task_meta,
        "transfer": transfer_meta,
        "isr_cov": isr_cov_meta,
        "isr_multiclass": isr_multi_meta,
        "vicreg": vicreg_meta,
    }


def read_causal(step3_dir, step4_dir):
    causal = {}
    p3 = Path(step3_dir) / "matched_random_comparison.csv"
    if p3.exists():
        rows = list(csv.DictReader(p3.open()))
        # Adapted model, EN, full removal.
        for r in rows:
            if (
                r["model_state"] == "adapted"
                and r["language"] == "en"
                and abs(float(r["strength"]) - 1.0) < 1e-12
            ):
                causal.setdefault(r["real_subspace"], {})[
                    "removal_excess_vs_random"
                ] = float(r["causal_excess_loss_vs_random"])

    p4 = Path(step4_dir) / "matched_random_rescue_comparison.csv"
    if p4.exists():
        rows = list(csv.DictReader(p4.open()))
        for r in rows:
            if abs(float(r["alpha"]) - 1.0) < 1e-12:
                d = causal.setdefault(r["real_subspace"], {})
                d["old_rescue_excess_vs_random"] = float(
                    r["old_rescue_excess_vs_random"]
                )
                d["new_language_cost"] = float(r["real_new_language_cost"])
                d["old_recovery_fraction"] = float(
                    r["real_old_recovery_fraction"]
                )
    return causal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--anchor_checkpoint",
        default="invariance_runs/sequence_seed0/en__zh/stage1_en",
    )
    ap.add_argument(
        "--adapted_checkpoint",
        default="mechanism_runs/step1_layer_lambda_sweep/checkpoints/full_ft",
    )
    ap.add_argument(
        "--languages",
        nargs="+",
        default=["en", "zh", "fr", "de", "es"],
    )
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16, 20, 24],
        help="Use 1..24 to build a full-depth atlas.",
    )
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--probe_train_per_lang", type=int, default=800)
    ap.add_argument("--probe_test_per_lang", type=int, default=800)
    ap.add_argument("--aligned_examples", type=int, default=1200)
    ap.add_argument("--extract_batch", type=int, default=16)
    ap.add_argument("--probe_epochs", type=int, default=60)
    ap.add_argument("--isr_cov_class", type=int, default=0)
    ap.add_argument("--vicreg_epochs", type=int, default=150)
    ap.add_argument("--alignment_max_examples", type=int, default=2000)
    ap.add_argument(
        "--step3_dir",
        default="mechanism_runs/step3_causal_removal_v2",
    )
    ap.add_argument(
        "--step4_dir",
        default="mechanism_runs/step4_causal_rescue_v2",
    )
    ap.add_argument(
        "--out_dir",
        default="mechanism_runs/functional_representation_atlas",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    feat_dir = out / "features"
    basis_dir = out / "bases"
    data_dir.mkdir(exist_ok=True)
    feat_dir.mkdir(exist_ok=True)
    basis_dir.mkdir(exist_ok=True)

    probe_file = data_dir / "xnli_probe.jsonl"
    aligned_file = data_dir / "xnli_aligned.jsonl"

    maybe(
        probe_file,
        [
            sys.executable,
            INV / "prepare_xnli.py",
            "--languages",
            *args.languages,
            "--train_per_lang",
            args.probe_train_per_lang,
            "--test_per_lang",
            args.probe_test_per_lang,
            "--seed",
            2027,
            "--out_file",
            probe_file,
        ],
        args.force,
    )
    maybe(
        aligned_file,
        [
            sys.executable,
            INV / "prepare_aligned_xnli.py",
            "--languages",
            *args.languages,
            "--split",
            "validation",
            "--n_examples",
            args.aligned_examples,
            "--seed",
            2027,
            "--out_file",
            aligned_file,
        ],
        args.force,
    )

    anchor_probe = feat_dir / "anchor_probe.pt"
    adapted_probe = feat_dir / "adapted_probe.pt"
    anchor_aligned = feat_dir / "anchor_aligned.pt"
    adapted_aligned = feat_dir / "adapted_aligned.pt"

    for ckpt, data_file, out_file in [
        (args.anchor_checkpoint, probe_file, anchor_probe),
        (args.adapted_checkpoint, probe_file, adapted_probe),
        (args.anchor_checkpoint, aligned_file, anchor_aligned),
        (args.adapted_checkpoint, aligned_file, adapted_aligned),
    ]:
        maybe(
            out_file,
            [
                sys.executable,
                INV / "extract_hidden.py",
                "--checkpoint",
                ckpt,
                "--data_file",
                data_file,
                "--out_file",
                out_file,
                "--layers",
                *args.layers,
                "--batch_size",
                args.extract_batch,
            ],
            args.force,
        )

    pa = torch.load(anchor_probe, map_location="cpu")
    pb = torch.load(adapted_probe, map_location="cpu")
    aa = torch.load(anchor_aligned, map_location="cpu")
    ab = torch.load(adapted_aligned, map_location="cpu")
    aligned_rows = read_jsonl(aligned_file)

    if pa["example_ids"] != pb["example_ids"]:
        raise ValueError("Anchor/adapted probe feature order mismatch.")
    if aa["example_ids"] != ab["example_ids"]:
        raise ValueError("Anchor/adapted aligned feature order mismatch.")

    languages = pa["languages"]
    unique_langs = sorted(set(languages))
    lang_to_id = {lang: i for i, lang in enumerate(unique_langs)}
    lang_ids = torch.tensor([lang_to_id[x] for x in languages], dtype=torch.long)
    labels = pa["labels"].long()
    train_idx = torch.tensor(
        [i for i, s in enumerate(pa["splits"]) if s == "probe_train"],
        dtype=torch.long,
    )
    test_idx = torch.tensor(
        [i for i, s in enumerate(pa["splits"]) if s == "probe_test"],
        dtype=torch.long,
    )

    causal = read_causal(args.step3_dir, args.step4_dir)

    layer_rows = []
    profile_rows = []
    overlap_rows = []

    for layer in args.layers:
        print(f"\n================ layer {layer} ================")
        xa = pa["features"][str(layer)].float()
        xb = pb["features"][str(layer)].float()
        xaa = aa["features"][str(layer)].float()
        xab = ab["features"][str(layer)].float()

        if xa.shape != xb.shape:
            raise ValueError(f"Layer {layer}: anchor/adapted probe shape mismatch.")

        d1, d2, dsym = linear_alignment_discrepancy(
            xa,
            xb,
            max_examples=args.alignment_max_examples,
        )
        layer_rows.append(
            {
                "layer": layer,
                "relative_l2_drift": relative_l2_drift(xa, xb),
                "linear_cka": linear_cka(xa, xb),
                "mean_example_cosine": mean_example_cosine(xa, xb),
                "alignment_discrepancy_adapted_to_anchor": d1,
                "alignment_discrepancy_anchor_to_adapted": d2,
                "alignment_discrepancy_symmetric": dsym,
            }
        )

        anchor_sub, anchor_meta = build_subspaces(
            xa,
            xaa,
            pa,
            aligned_rows,
            rank=args.rank,
            probe_epochs=args.probe_epochs,
            isr_cov_class=args.isr_cov_class,
            vicreg_epochs=args.vicreg_epochs,
            seed_offset=layer * 100,
        )
        adapted_sub, adapted_meta = build_subspaces(
            xb,
            xab,
            pb,
            aligned_rows,
            rank=args.rank,
            probe_epochs=args.probe_epochs,
            isr_cov_class=args.isr_cov_class,
            vicreg_epochs=args.vicreg_epochs,
            seed_offset=50000 + layer * 100,
        )

        q_drift, drift_meta = fit_drift_basis(xa, xb, args.rank)
        anchor_sub["drift"] = q_drift

        delta = xb - xa

        torch.save(
            {
                "layer": layer,
                "anchor_subspaces": anchor_sub,
                "adapted_subspaces": adapted_sub,
                "anchor_metadata": anchor_meta,
                "adapted_metadata": adapted_meta,
                "drift_metadata": drift_meta,
            },
            basis_dir / f"layer{layer}.pt",
        )

        for name, q in anchor_sub.items():
            drift_frac, drift_enrichment = projected_drift_metrics(delta, q)

            if name == "drift":
                stability = {
                    "projection_overlap": float("nan"),
                    "mean_cos2": float("nan"),
                    "mean_angle_deg": float("nan"),
                    "max_angle_deg": float("nan"),
                    "min_rank": int(q.shape[1]),
                }
            else:
                stability = principal_stats(q, adapted_sub[name])

            lang_acc_anchor = probe_accuracy_in_subspace(
                xa,
                q,
                lang_ids,
                train_idx,
                test_idx,
                len(unique_langs),
                epochs=args.probe_epochs,
                seed=90000 + layer,
            )
            task_acc_anchor = probe_accuracy_in_subspace(
                xa,
                q,
                labels,
                train_idx,
                test_idx,
                int(labels.max().item()) + 1,
                epochs=args.probe_epochs,
                seed=91000 + layer,
            )

            # Readability after adaptation in the SAME anchor-defined basis.
            lang_acc_adapted = probe_accuracy_in_subspace(
                xb,
                q,
                lang_ids,
                train_idx,
                test_idx,
                len(unique_langs),
                epochs=args.probe_epochs,
                seed=92000 + layer,
            )
            task_acc_adapted = probe_accuracy_in_subspace(
                xb,
                q,
                labels,
                train_idx,
                test_idx,
                int(labels.max().item()) + 1,
                epochs=args.probe_epochs,
                seed=93000 + layer,
            )

            row = {
                "layer": layer,
                "subspace": name,
                "rank": int(q.shape[1]),
                "projected_drift_fraction": drift_frac,
                "projected_drift_enrichment": drift_enrichment,
                "subspace_stability_overlap": stability["projection_overlap"],
                "subspace_stability_mean_angle_deg": stability["mean_angle_deg"],
                "subspace_stability_max_angle_deg": stability["max_angle_deg"],
                "anchor_language_probe_accuracy": lang_acc_anchor,
                "adapted_language_probe_accuracy_in_anchor_basis": lang_acc_adapted,
                "anchor_task_probe_accuracy": task_acc_anchor,
                "adapted_task_probe_accuracy_in_anchor_basis": task_acc_adapted,
                "language_probe_change": lang_acc_adapted - lang_acc_anchor,
                "task_probe_change": task_acc_adapted - task_acc_anchor,
            }

            # Causal columns are currently available only for the detailed L20
            # Step-2 names. Merge when names match.
            if layer == 20 and name in causal:
                row.update(causal[name])
            profile_rows.append(row)

        names = list(anchor_sub)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                st = principal_stats(anchor_sub[a], anchor_sub[b])
                overlap_rows.append(
                    {
                        "layer": layer,
                        "subspace_a": a,
                        "subspace_b": b,
                        "projection_overlap": st["projection_overlap"],
                        "mean_angle_deg": st["mean_angle_deg"],
                        "max_angle_deg": st["max_angle_deg"],
                        "min_rank": st["min_rank"],
                    }
                )

        print(
            f"layer={layer} relL2={layer_rows[-1]['relative_l2_drift']:.5f} "
            f"CKA={layer_rows[-1]['linear_cka']:.5f} "
            f"alignD={layer_rows[-1]['alignment_discrepancy_symmetric']:.5f}"
        )
        for r in [x for x in profile_rows if x["layer"] == layer]:
            print(
                f"  {r['subspace']:16s} "
                f"drift={r['projected_drift_fraction']:.4f} "
                f"enrich={r['projected_drift_enrichment']:.2f} "
                f"stable={r['subspace_stability_overlap']:.4f} "
                f"lang={r['anchor_language_probe_accuracy']:.3f} "
                f"task={r['anchor_task_probe_accuracy']:.3f}"
            )

    def write_csv(path, rows):
        fields = sorted({k for r in rows for k in r.keys()})
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write_csv(out / "layer_representation_metrics.csv", layer_rows)
    write_csv(out / "functional_subspace_profile.csv", profile_rows)
    write_csv(out / "subspace_overlap_by_layer.csv", overlap_rows)

    # A concise layer x functional-subspace matrix for projected drift.
    matrix = {}
    for r in profile_rows:
        matrix.setdefault(str(r["layer"]), {})[r["subspace"]] = {
            "drift_fraction": r["projected_drift_fraction"],
            "drift_enrichment": r["projected_drift_enrichment"],
            "stability_overlap": r["subspace_stability_overlap"],
            "language_probe_accuracy": r["anchor_language_probe_accuracy"],
            "task_probe_accuracy": r["anchor_task_probe_accuracy"],
        }

    manifest = {
        "anchor_checkpoint": args.anchor_checkpoint,
        "adapted_checkpoint": args.adapted_checkpoint,
        "languages": args.languages,
        "layers": args.layers,
        "rank": args.rank,
        "definitions": {
            "alignment_discrepancy": (
                "minimum centered linear-alignment residual normalized by "
                "reference Frobenius energy; symmetric score averages both directions"
            ),
            "projected_drift_fraction": (
                "||DeltaH Q||_F^2 / ||DeltaH||_F^2 using anchor-defined Q"
            ),
            "projected_drift_enrichment": (
                "projected drift fraction divided by rank/hidden_dim; >1 means "
                "drift concentrates in the subspace more than isotropic expectation"
            ),
            "subspace_stability_overlap": (
                "mean squared cosine of principal angles between anchor- and "
                "adapted-fitted subspaces"
            ),
            "probe_information": (
                "fresh linear classifier accuracy using only coordinates in the "
                "specified anchor-defined subspace"
            ),
        },
        "causal_step3_dir": args.step3_dir,
        "causal_step4_dir": args.step4_dir,
    }
    (out / "atlas.json").write_text(json.dumps(matrix, indent=2))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nSaved:")
    print(out / "layer_representation_metrics.csv")
    print(out / "functional_subspace_profile.csv")
    print(out / "subspace_overlap_by_layer.csv")
    print(out / "atlas.json")


if __name__ == "__main__":
    main()
