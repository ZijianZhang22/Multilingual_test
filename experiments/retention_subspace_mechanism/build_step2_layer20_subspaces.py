#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
INV = ROOT / "invariance"

from experiments.retention_subspace_mechanism.subspace_extractors import (  # noqa: E402
    fit_isr_cov,
    fit_isr_multiclass_semantic,
    fit_vicreg_linear,
    orthonormal_random,
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


def fit_drift_basis(anchor_features, adapted_features, layer, rank):
    a = torch.load(anchor_features, map_location="cpu")
    b = torch.load(adapted_features, map_location="cpu")
    xa = a["features"][str(layer)].float()
    xb = b["features"][str(layer)].float()
    if xa.shape != xb.shape:
        raise ValueError(f"Feature shape mismatch: {xa.shape} vs {xb.shape}")
    if a["example_ids"] != b["example_ids"]:
        raise ValueError("Anchor/adapted feature example order mismatch.")
    delta = xb - xa
    delta_centered = delta - delta.mean(dim=0, keepdim=True)
    q = min(rank, delta_centered.shape[0], delta_centered.shape[1])
    _, s, v = torch.pca_lowrank(delta_centered, q=q, center=False)
    basis = v[:, :q]
    explained = (s[:q] ** 2)
    explained = explained / explained.sum().clamp_min(1e-12)
    return basis, delta, explained


def overlap_stats(qa, qb):
    s = torch.linalg.svdvals(qa.T @ qb)
    return {
        "mean_squared_cosine": float((s ** 2).mean()),
        "max_cosine": float(s.max()),
        "mean_principal_angle_deg": float(
            torch.rad2deg(torch.acos(s.clamp(-1, 1))).mean()
        ),
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Step 2: construct multiple Layer-20 multilingual mechanism subspaces "
            "(INLP, transfer PCA, drift PCA, ISR-Cov, ISR-Multiclass-derived, "
            "VICReg-linear) plus matched-rank random controls."
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
        "--languages",
        nargs="+",
        default=["en", "zh", "fr", "de", "es"],
        help=(
            "Languages used to FIT multilingual subspaces. They need not all be "
            "training languages; XNLI languages give multiple environments."
        ),
    )
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--inlp_iters", type=int, default=32)
    ap.add_argument("--probe_train_per_lang", type=int, default=1200)
    ap.add_argument("--probe_test_per_lang", type=int, default=1200)
    ap.add_argument("--aligned_examples", type=int, default=2000)
    ap.add_argument("--extract_batch", type=int, default=16)
    ap.add_argument(
        "--isr_cov_class",
        type=int,
        default=0,
        help="XNLI class used for class-conditional ISR-Cov (default entailment=0).",
    )
    ap.add_argument("--vicreg_epochs", type=int, default=300)
    ap.add_argument("--vicreg_lr", type=float, default=3e-2)
    ap.add_argument("--out_dir", default="mechanism_runs/step2_layer20_subspaces")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    feat_dir = out / "features"
    data_dir.mkdir(exist_ok=True)
    feat_dir.mkdir(exist_ok=True)

    probe = data_dir / "xnli_probe.jsonl"
    aligned = data_dir / "xnli_aligned.jsonl"
    anchor_probe = feat_dir / "anchor_probe.pt"
    adapted_probe = feat_dir / "adapted_probe.pt"
    anchor_aligned = feat_dir / "anchor_aligned.pt"
    lang_file = out / "language_inlp.pt"
    transfer_file = out / f"transfer_rank{args.rank}.pt"

    maybe(
        probe,
        [
            sys.executable, INV / "prepare_xnli.py",
            "--languages", *args.languages,
            "--train_per_lang", args.probe_train_per_lang,
            "--test_per_lang", args.probe_test_per_lang,
            "--seed", 2026,
            "--out_file", probe,
        ],
        args.force,
    )
    maybe(
        aligned,
        [
            sys.executable, INV / "prepare_aligned_xnli.py",
            "--languages", *args.languages,
            "--split", "validation",
            "--n_examples", args.aligned_examples,
            "--seed", 2026,
            "--out_file", aligned,
        ],
        args.force,
    )

    for ckpt, data_file, out_file in [
        (args.anchor_checkpoint, probe, anchor_probe),
        (args.adapted_checkpoint, probe, adapted_probe),
        (args.anchor_checkpoint, aligned, anchor_aligned),
    ]:
        maybe(
            out_file,
            [
                sys.executable, INV / "extract_hidden.py",
                "--checkpoint", ckpt,
                "--data_file", data_file,
                "--out_file", out_file,
                "--layers", args.layer,
                "--batch_size", args.extract_batch,
            ],
            args.force,
        )

    maybe(
        lang_file,
        [
            sys.executable, INV / "fit_inlp_language_subspace.py",
            "--features_file", anchor_probe,
            "--out_file", lang_file,
            "--layer", args.layer,
            "--iters", args.inlp_iters,
            "--seed", 0,
        ],
        args.force,
    )
    maybe(
        transfer_file,
        [
            sys.executable, INV / "fit_transferable_subspace.py",
            "--features_file", anchor_aligned,
            "--aligned_data_file", aligned,
            "--language_subspace_file", lang_file,
            "--out_file", transfer_file,
            "--layer", args.layer,
            "--rank", args.rank,
        ],
        args.force,
    )

    probe_payload = torch.load(anchor_probe, map_location="cpu")
    aligned_payload = torch.load(anchor_aligned, map_location="cpu")
    x_probe = probe_payload["features"][str(args.layer)].float()
    x_aligned = aligned_payload["features"][str(args.layer)].float()
    center = x_probe.mean(dim=0)

    lang = torch.load(lang_file, map_location="cpu")
    q_lang = lang["language_subspace_basis"].float()
    transfer = torch.load(transfer_file, map_location="cpu")
    q_transfer = transfer["transferable_subspace_basis"].float()

    q_drift, delta, drift_ev = fit_drift_basis(
        anchor_probe, adapted_probe, args.layer, args.rank
    )

    # Use probe_train only for environment/class-based subspace fitting.
    train_idx = torch.tensor(
        [i for i, s in enumerate(probe_payload["splits"]) if s == "probe_train"],
        dtype=torch.long,
    )
    x_train = x_probe[train_idx]
    langs_train = [probe_payload["languages"][i] for i in train_idx.tolist()]
    labels_train = probe_payload["labels"][train_idx].long()

    q_isr_cov, isr_cov_meta = fit_isr_cov(
        x_train,
        langs_train,
        labels_train,
        rank=args.rank,
        class_label=args.isr_cov_class,
    )
    q_isr_multi, q_isr_spurious, isr_multi_meta = fit_isr_multiclass_semantic(
        x_train,
        langs_train,
        labels_train,
        rank=args.rank,
    )

    aligned_rows = read_jsonl(aligned)
    pair_ids = [r["pair_id"] for r in aligned_rows]
    if len(pair_ids) != x_aligned.shape[0]:
        raise ValueError("Aligned JSONL/features length mismatch for VICReg.")
    q_vicreg, vicreg_meta = fit_vicreg_linear(
        x_aligned,
        pair_ids,
        rank=args.rank,
        epochs=args.vicreg_epochs,
        lr=args.vicreg_lr,
        seed=0,
    )

    dim = x_probe.shape[1]
    real_subspaces = {
        "language": q_lang,
        "transfer": q_transfer,
        "drift": q_drift,
        "isr_cov": q_isr_cov,
        "isr_multiclass": q_isr_multi,
        "vicreg": q_vicreg,
    }

    subspaces = {}
    controls = {}
    for i, (name, q) in enumerate(real_subspaces.items()):
        subspaces[name] = q
        random_name = f"random_{name}"
        subspaces[random_name] = orthonormal_random(
            dim, q.shape[1], seed=4101 + i
        )
        controls[name] = random_name

    # Pairwise overlap across every REAL candidate mechanism.
    overlaps = {}
    names = list(real_subspaces)
    for i, a in enumerate(names):
        for b in names[i:]:
            overlaps[f"{a}__{b}"] = overlap_stats(
                real_subspaces[a], real_subspaces[b]
            )

    definitions = {
        "language": (
            "INLP language-decodable directions fitted on anchor XNLI features."
        ),
        "transfer": (
            "PCA of aligned cross-language semantic means after removing the "
            "INLP language subspace."
        ),
        "drift": (
            "Top PCA directions of adapted-minus-anchor Layer-20 feature deltas."
        ),
        "isr_cov": (
            "ISR-Cov-inspired class-conditional covariance-invariant directions "
            "aggregated across language pairs with a projection/flag mean."
        ),
        "isr_multiclass": (
            "ISR-Multiclass-derived invariant semantic basis: exact multiclass "
            "environment-varying mean directions are removed, then semantic PCA "
            "is taken in the resulting invariant nullspace."
        ),
        "vicreg": (
            "VICReg-inspired orthonormal linear cross-lingual subspace learned "
            "from aligned multilingual XNLI views."
        ),
        "random_*": (
            "Seeded isotropic orthonormal basis matched in rank to each named "
            "real subspace."
        ),
    }

    artifact = {
        "layer": args.layer,
        "hidden_dim": dim,
        "anchor_checkpoint": args.anchor_checkpoint,
        "adapted_checkpoint": args.adapted_checkpoint,
        "fit_languages": args.languages,
        "center": center,
        "subspaces": subspaces,
        "real_subspaces": list(real_subspaces),
        "matched_random_controls": controls,
        "ranks": {k: int(v.shape[1]) for k, v in subspaces.items()},
        "definitions": definitions,
        "drift_mean_delta_l2": float(delta.norm(dim=1).mean()),
        "drift_explained_fraction_within_rank": drift_ev,
        "isr_cov_metadata": isr_cov_meta,
        "isr_multiclass_metadata": isr_multi_meta,
        "isr_multiclass_spurious_basis": q_isr_spurious,
        "vicreg_metadata": vicreg_meta,
        "overlaps": overlaps,
        "probe_file": str(probe),
        "aligned_file": str(aligned),
    }
    out_file = out / "layer20_subspaces.pt"
    torch.save(artifact, out_file)

    summary = {
        "layer": args.layer,
        "hidden_dim": dim,
        "fit_languages": args.languages,
        "real_subspaces": artifact["real_subspaces"],
        "matched_random_controls": controls,
        "ranks": artifact["ranks"],
        "drift_mean_delta_l2": artifact["drift_mean_delta_l2"],
        "isr_cov_metadata": isr_cov_meta,
        "isr_multiclass_metadata": isr_multi_meta,
        "vicreg_metadata": vicreg_meta,
        "overlaps": overlaps,
        "definitions": definitions,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Layer-20 real subspaces ===")
    for name in artifact["real_subspaces"]:
        q = subspaces[name]
        print(f"{name:18s} rank={q.shape[1]}")
    print("\n=== Matched random controls ===")
    for real, random_name in controls.items():
        print(
            f"{real:18s} -> {random_name:24s} "
            f"rank={subspaces[random_name].shape[1]}"
        )
    print("\n=== Selected real-subspace overlaps ===")
    for k, v in overlaps.items():
        if "__" in k and not k.split("__")[0] == k.split("__")[1]:
            print(
                f"{k:30s} mean_cos2={v['mean_squared_cosine']:.6f} "
                f"max_cos={v['max_cosine']:.6f}"
            )
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
