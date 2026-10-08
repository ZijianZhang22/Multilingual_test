#!/usr/bin/env python3
"""Compare mean-pooled and last-token fitted subspaces.

The previous behavioral intervention used bases fitted on mean-pooled features
but edited only the last prompt token. This script quantifies how serious that
feature-location mismatch is by comparing principal-angle overlap between
subspaces fitted with the two pooling choices.
"""

import argparse
import csv
import json
from pathlib import Path

import torch


def orth(q):
    q, _ = torch.linalg.qr(q.float(), mode="reduced")
    return q


def stats(a, b):
    qa, qb = orth(a), orth(b)
    s = torch.linalg.svdvals(qa.T @ qb).clamp(0, 1)
    r = min(qa.shape[1], qb.shape[1])
    angles = torch.rad2deg(torch.acos(s))
    return {
        "rank_a": int(qa.shape[1]),
        "rank_b": int(qb.shape[1]),
        "projection_overlap": float(s.pow(2).sum() / max(r, 1)),
        "mean_cos2": float(s.pow(2).mean()),
        "mean_principal_angle_deg": float(angles.mean()),
        "max_principal_angle_deg": float(angles.max()),
        "min_cosine": float(s.min()),
        "max_cosine": float(s.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean_core", required=True)
    ap.add_argument("--last_core", required=True)
    ap.add_argument(
        "--out_dir",
        default="mechanism_runs/semantic_causal_validation/pooling_comparison",
    )
    args = ap.parse_args()

    a = torch.load(args.mean_core, map_location="cpu")
    b = torch.load(args.last_core, map_location="cpu")
    if int(a["hidden_dim"]) != int(b["hidden_dim"]):
        raise ValueError("Hidden dimensions differ.")
    if int(a["layer"]) != int(b["layer"]):
        raise ValueError("Layer numbers differ.")

    names = sorted(
        set(a.get("real_subspaces", [])) & set(b.get("real_subspaces", []))
    )
    rows = []
    for name in names:
        row = {"subspace": name, **stats(a["subspaces"][name], b["subspaces"][name])}
        rows.append(row)
        print(
            f"{name:16s} overlap={row['projection_overlap']:.4f} "
            f"angle={row['mean_principal_angle_deg']:.1f}deg",
            flush=True,
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "pooling_subspace_overlap.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    (out / "manifest.json").write_text(
        json.dumps(
            {
                **vars(args),
                "layer": int(a["layer"]),
                "hidden_dim": int(a["hidden_dim"]),
                "mean_pool_declared": a.get("pool", "legacy/unknown"),
                "last_pool_declared": b.get("pool", "legacy/unknown"),
                "interpretation": (
                    "Low overlap means mean-pooled bases should not be treated as "
                    "equivalent to last-token intervention directions."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
