import argparse
import csv
from pathlib import Path

import torch


def squared_norm_rows(x):
    return (x.float() * x.float()).sum(dim=1)


def metrics(delta, q):
    total_e = squared_norm_rows(delta)
    sub_e = squared_norm_rows(delta @ q)
    residual_e = (total_e - sub_e).clamp_min(0.0)

    d = delta.shape[1]
    k = q.shape[1]
    r = max(d - k, 1)

    return {
        "total_drift_l2": float(total_e.mean().sqrt().item()),
        "language_subspace_drift_l2": float(sub_e.mean().sqrt().item()),
        "shared_residual_drift_l2": float(residual_e.mean().sqrt().item()),
        "language_drift_per_dim": float(sub_e.mean().item()) / max(k, 1),
        "shared_drift_per_dim": float(residual_e.mean().item()) / r,
        "language_drift_fraction": (
            float(sub_e.sum().item()) / float(total_e.sum().item())
            if float(total_e.sum().item()) > 0 else 0.0
        ),
        "shared_drift_fraction": (
            float(residual_e.sum().item()) / float(total_e.sum().item())
            if float(total_e.sum().item()) > 0 else 0.0
        ),
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compare one old-language anchor checkpoint with the post-adaptation "
            "checkpoint in a fixed anchor-fitted INLP language subspace."
        )
    )
    ap.add_argument("--anchor_features", required=True)
    ap.add_argument("--post_features", required=True)
    ap.add_argument("--language_subspace_file", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    a = torch.load(args.anchor_features, map_location="cpu")
    p = torch.load(args.post_features, map_location="cpu")
    qd = torch.load(args.language_subspace_file, map_location="cpu")

    layer = str(args.layer)
    if layer not in a["features"] or layer not in p["features"]:
        raise ValueError(f"Layer {args.layer} missing from one feature file")

    for key in ["languages", "splits", "example_ids"]:
        if a.get(key) != p.get(key):
            raise ValueError(f"Anchor/post mismatch in {key}")
    if not torch.equal(a["labels"], p["labels"]):
        raise ValueError("Anchor/post labels do not align")

    q = qd["language_subspace_basis"].float()
    if int(qd["layer"]) != args.layer:
        raise ValueError("Subspace layer mismatch")

    xa = a["features"][layer].float()
    xp = p["features"][layer].float()
    delta = xp - xa

    rows = []
    langs = a["languages"]
    splits = a["splits"]

    # Overall probe-test drift.
    all_idx = torch.tensor(
        [i for i, s in enumerate(splits) if s == "probe_test"], dtype=torch.long
    )
    rows.append({
        "language": "__all__",
        "n": int(all_idx.numel()),
        **metrics(delta[all_idx], q),
    })

    for lang in sorted(set(langs)):
        idx = torch.tensor(
            [
                i for i, (l, s) in enumerate(zip(langs, splits))
                if l == lang and s == "probe_test"
            ],
            dtype=torch.long,
        )
        rows.append({
            "language": lang,
            "n": int(idx.numel()),
            **metrics(delta[idx], q),
        })

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n=== Anchor -> post representation drift ===")
    for r in rows:
        print(
            f"{r['language']:<8} n={r['n']:<5} "
            f"total={r['total_drift_l2']:.4f} "
            f"lang={r['language_subspace_drift_l2']:.4f} "
            f"shared={r['shared_residual_drift_l2']:.4f} "
            f"lang/dim={r['language_drift_per_dim']:.6f} "
            f"shared/dim={r['shared_drift_per_dim']:.6f}"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
