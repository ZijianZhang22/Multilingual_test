#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
INV = ROOT / "invariance"


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


def orthonormal_random(dim, rank, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(dim, rank, generator=g)
    q, _ = torch.linalg.qr(q, mode="reduced")
    return q[:, :rank]


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


def main():
    ap = argparse.ArgumentParser(
        description="Step 2: construct Layer-20 language, transfer, drift, and matched-random subspaces."
    )
    ap.add_argument("--anchor_checkpoint", default="invariance_runs/sequence_seed0/en__zh/stage1_en")
    ap.add_argument("--adapted_checkpoint", default="mechanism_runs/step1_layer_lambda_sweep/checkpoints/full_ft")
    ap.add_argument("--languages", nargs="+", default=["en", "zh"])
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--transfer_rank", type=int, default=64)
    ap.add_argument("--drift_rank", type=int, default=64)
    ap.add_argument("--inlp_iters", type=int, default=32)
    ap.add_argument("--probe_train_per_lang", type=int, default=1200)
    ap.add_argument("--probe_test_per_lang", type=int, default=1200)
    ap.add_argument("--aligned_examples", type=int, default=2000)
    ap.add_argument("--extract_batch", type=int, default=16)
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
    transfer_file = out / f"transfer_rank{args.transfer_rank}.pt"

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
            "--rank", args.transfer_rank,
        ],
        args.force,
    )

    anchor_payload = torch.load(anchor_probe, map_location="cpu")
    x_anchor = anchor_payload["features"][str(args.layer)].float()
    center = x_anchor.mean(dim=0)

    lang = torch.load(lang_file, map_location="cpu")
    q_lang = lang["language_subspace_basis"].float()
    transfer = torch.load(transfer_file, map_location="cpu")
    q_transfer = transfer["transferable_subspace_basis"].float()

    q_drift, delta, drift_ev = fit_drift_basis(
        anchor_probe, adapted_probe, args.layer, args.drift_rank
    )

    dim = x_anchor.shape[1]
    subspaces = {
        "language": q_lang,
        "transfer": q_transfer,
        "drift": q_drift,
        "random_language": orthonormal_random(dim, q_lang.shape[1], 4101),
        "random_transfer": orthonormal_random(dim, q_transfer.shape[1], 4102),
        "random_drift": orthonormal_random(dim, q_drift.shape[1], 4103),
    }

    # Basic overlap diagnostics. Large overlap means causal interpretations
    # should account for non-independence between candidate mechanisms.
    overlaps = {}
    real = ["language", "transfer", "drift"]
    for a in real:
        for b in real:
            qa, qb = subspaces[a], subspaces[b]
            s = torch.linalg.svdvals(qa.T @ qb)
            overlaps[f"{a}__{b}"] = {
                "mean_squared_cosine": float((s**2).mean()),
                "max_cosine": float(s.max()),
            }

    artifact = {
        "layer": args.layer,
        "hidden_dim": dim,
        "anchor_checkpoint": args.anchor_checkpoint,
        "adapted_checkpoint": args.adapted_checkpoint,
        "center": center,
        "subspaces": subspaces,
        "ranks": {k: int(v.shape[1]) for k, v in subspaces.items()},
        "definitions": {
            "language": "INLP language-decodable directions fitted on anchor XNLI features.",
            "transfer": "PCA of aligned cross-language semantic means after removing language subspace.",
            "drift": "Top PCA directions of adapted-minus-anchor Layer-20 feature deltas.",
            "random_*": "Seeded isotropic orthonormal basis matched in rank to the named real subspace.",
        },
        "drift_mean_delta_l2": float(delta.norm(dim=1).mean()),
        "drift_explained_fraction_within_rank": drift_ev,
        "overlaps": overlaps,
        "probe_file": str(probe),
        "aligned_file": str(aligned),
    }
    out_file = out / "layer20_subspaces.pt"
    torch.save(artifact, out_file)

    summary = {
        "layer": args.layer,
        "hidden_dim": dim,
        "ranks": artifact["ranks"],
        "drift_mean_delta_l2": artifact["drift_mean_delta_l2"],
        "overlaps": overlaps,
        "definitions": artifact["definitions"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Layer-20 subspaces ===")
    for name, q in subspaces.items():
        print(f"{name:18s} rank={q.shape[1]}")
    print("\n=== Real-subspace overlap ===")
    for k, v in overlaps.items():
        print(
            f"{k:20s} mean_cos2={v['mean_squared_cosine']:.6f} "
            f"max_cos={v['max_cosine']:.6f}"
        )
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
