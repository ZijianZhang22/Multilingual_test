import argparse
import csv
from itertools import combinations
from pathlib import Path

import torch


def orthonormalize(q):
    q = q.float()
    q, _ = torch.linalg.qr(q, mode="reduced")
    return q


def load_basis(path, preferred_keys):
    obj = torch.load(path, map_location="cpu")
    for key in preferred_keys:
        if key in obj:
            return orthonormalize(obj[key]), obj
    raise KeyError(f"{path} contains none of {preferred_keys}")


def principal_angle_metrics(q1, q2):
    q1 = orthonormalize(q1)
    q2 = orthonormalize(q2)
    s = torch.linalg.svdvals(q1.T @ q2).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(s))
    overlap = float((s * s).sum().item()) / max(min(q1.shape[1], q2.shape[1]), 1)
    return {
        "rank_a": int(q1.shape[1]),
        "rank_b": int(q2.shape[1]),
        "principal_angle_min_deg": float(angles.min().item()),
        "principal_angle_mean_deg": float(angles.mean().item()),
        "principal_angle_max_deg": float(angles.max().item()),
        "normalized_projection_overlap": overlap,
    }


def fit_drift_basis(anchor_features, post_features, layer, rank, old_language=None):
    a = torch.load(anchor_features, map_location="cpu")
    p = torch.load(post_features, map_location="cpu")
    for key in ["languages", "splits", "example_ids"]:
        if a.get(key) != p.get(key):
            raise ValueError(f"Anchor/post mismatch in {key}")
    xa = a["features"][str(layer)].float()
    xp = p["features"][str(layer)].float()
    idx = torch.tensor(
        [
            i for i, (lang, split) in enumerate(zip(a["languages"], a["splits"]))
            if split == "probe_test" and (old_language is None or lang == old_language)
        ],
        dtype=torch.long,
    )
    delta = xp[idx] - xa[idx]
    delta = delta - delta.mean(dim=0, keepdim=True)
    q = min(rank, delta.shape[0], delta.shape[1])
    _, _, v = torch.pca_lowrank(delta, q=q, center=False)
    return orthonormalize(v[:, :q])


def main():
    ap = argparse.ArgumentParser(
        description="Compare functional representation subspaces with principal angles."
    )
    ap.add_argument("--language_subspace_file", required=True)
    ap.add_argument("--task_subspace_file", default=None)
    ap.add_argument("--transferable_subspace_file", default=None)
    ap.add_argument("--anchor_features", default=None)
    ap.add_argument("--post_features", default=None)
    ap.add_argument("--old_language", default=None)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--drift_rank", type=int, default=32)
    ap.add_argument("--drift_subspace_out", default=None)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    bases = {}
    q_lang, _ = load_basis(
        args.language_subspace_file, ["language_subspace_basis"]
    )
    bases["language"] = q_lang

    if args.task_subspace_file:
        q_task, _ = load_basis(
            args.task_subspace_file,
            ["task_sensitive_subspace_basis", "gradient_subspace_basis"],
        )
        bases["task_sensitive"] = q_task

    if args.transferable_subspace_file:
        path = Path(args.transferable_subspace_file)
        if path.exists():
            q_transfer, _ = load_basis(path, ["transferable_subspace_basis"])
            bases["transferable"] = q_transfer

    if args.anchor_features and args.post_features:
        q_drift = fit_drift_basis(
            args.anchor_features,
            args.post_features,
            args.layer,
            args.drift_rank,
            old_language=args.old_language,
        )
        bases["drift"] = q_drift
        if args.drift_subspace_out:
            out_drift = Path(args.drift_subspace_out)
            out_drift.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "layer": args.layer,
                "rank": int(q_drift.shape[1]),
                "drift_subspace_basis": q_drift,
                "old_language": args.old_language,
                "anchor_features": args.anchor_features,
                "post_features": args.post_features,
            }, out_drift)

    dims = {name: q.shape[0] for name, q in bases.items()}
    if len(set(dims.values())) != 1:
        raise ValueError(f"Subspace ambient dimensions differ: {dims}")

    rows = []
    for (name_a, q_a), (name_b, q_b) in combinations(bases.items(), 2):
        rows.append({
            "subspace_a": name_a,
            "subspace_b": name_b,
            **principal_angle_metrics(q_a, q_b),
        })

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Principal-angle / overlap analysis ===")
    for r in rows:
        print(
            f"{r['subspace_a']:<14} vs {r['subspace_b']:<14} "
            f"mean_angle={r['principal_angle_mean_deg']:.2f}deg "
            f"overlap={r['normalized_projection_overlap']:.4f}"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
