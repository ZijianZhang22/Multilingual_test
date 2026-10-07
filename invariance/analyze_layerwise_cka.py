import argparse
import csv
from pathlib import Path

import torch


def ensure_aligned(a, b):
    for key in ["languages", "splits", "example_ids"]:
        if a.get(key) != b.get(key):
            raise ValueError(f"Feature-file mismatch in {key}")
    if not torch.equal(a["labels"], b["labels"]):
        raise ValueError("Feature-file labels do not align")


def center(x):
    return x.float() - x.float().mean(dim=0, keepdim=True)


def linear_cka(x, y, eps=1e-12):
    x = center(x)
    y = center(y)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    num = (cross * cross).sum()
    den = torch.sqrt((xx * xx).sum() * (yy * yy).sum()).clamp_min(eps)
    return float((num / den).item())


def relative_l2(x, y, eps=1e-12):
    delta = (y.float() - x.float()).pow(2).sum(dim=1).sqrt().mean()
    base = x.float().pow(2).sum(dim=1).sqrt().mean().clamp_min(eps)
    return float((delta / base).item())


def main():
    ap = argparse.ArgumentParser(
        description="Compute centered linear CKA between aligned checkpoint features."
    )
    ap.add_argument("--anchor_features", required=True)
    ap.add_argument("--post_features", required=True)
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    a = torch.load(args.anchor_features, map_location="cpu")
    p = torch.load(args.post_features, map_location="cpu")
    ensure_aligned(a, p)

    common = sorted(set(a["features"]) & set(p["features"]), key=lambda z: int(z))
    if args.layers:
        requested = {str(x) for x in args.layers}
        common = [x for x in common if x in requested]
    if not common:
        raise ValueError("No common requested layers in the two feature files")

    splits = a["splits"]
    languages = a["languages"]
    groups = ["__all__"] + sorted(set(languages))
    rows = []

    for layer in common:
        xa = a["features"][layer].float()
        xp = p["features"][layer].float()
        for lang in groups:
            idx = torch.tensor(
                [
                    i for i, (l, s) in enumerate(zip(languages, splits))
                    if s == "probe_test" and (lang == "__all__" or l == lang)
                ],
                dtype=torch.long,
            )
            rows.append({
                "layer": int(layer),
                "language": lang,
                "n": int(idx.numel()),
                "linear_cka": linear_cka(xa[idx], xp[idx]),
                "relative_l2_drift": relative_l2(xa[idx], xp[idx]),
            })

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Layerwise CKA ===")
    for r in rows:
        if r["language"] == "__all__":
            print(
                f"layer={r['layer']:>2} CKA={r['linear_cka']:.4f} "
                f"rel_L2={r['relative_l2_drift']:.4f}"
            )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
