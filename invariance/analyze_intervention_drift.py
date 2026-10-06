import argparse
import csv
from pathlib import Path

import torch


def drift(delta, q):
    coords = delta @ q
    return {
        "rms_coord": float(torch.sqrt(coords.float().pow(2).mean()).item()),
        "energy": float(coords.float().pow(2).sum().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor_features", required=True)
    ap.add_argument("--intervention_features", nargs="+", required=True)
    ap.add_argument("--language_subspace_file", required=True)
    ap.add_argument("--transferable_subspace_file", default=None)
    ap.add_argument("--out_file", default="invariance_analysis/intervention_drift.csv")
    ap.add_argument("--layer", type=int, default=12)
    args = ap.parse_args()

    anchor = torch.load(args.anchor_features, map_location="cpu")
    x0 = anchor["features"][str(args.layer)].float()

    q_lang = torch.load(
        args.language_subspace_file, map_location="cpu"
    )["language_subspace_basis"].float()

    q_transfer = None
    if args.transferable_subspace_file:
        q_transfer = torch.load(
            args.transferable_subspace_file, map_location="cpu"
        )["transferable_subspace_basis"].float()

    rows = []
    for path in args.intervention_features:
        cur = torch.load(path, map_location="cpu")
        x = cur["features"][str(args.layer)].float()
        if x.shape != x0.shape:
            raise ValueError(f"Shape mismatch for {path}")
        delta = x - x0

        lang = drift(delta, q_lang)
        lang_proj = (delta @ q_lang) @ q_lang.T
        shared = delta - lang_proj

        row = {
            "features_file": path,
            "checkpoint": cur.get("checkpoint", ""),
            "total_rms": float(torch.sqrt(delta.pow(2).mean()).item()),
            "lang_rms_coord": lang["rms_coord"],
            "shared_rms_per_hidden_dim": float(
                torch.sqrt(shared.pow(2).mean()).item()
            ),
        }
        if q_transfer is not None:
            row["transfer64_rms_coord"] = drift(delta, q_transfer)["rms_coord"]
        rows.append(row)

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {out}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
