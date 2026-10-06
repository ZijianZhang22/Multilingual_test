import argparse
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_file", required=True)
    ap.add_argument("--aligned_data_file", required=True)
    ap.add_argument("--language_subspace_file", required=True)
    ap.add_argument("--out_file", default="invariance_analysis/transferable64.pt")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--rank", type=int, default=64)
    args = ap.parse_args()

    payload = torch.load(args.features_file, map_location="cpu")
    x = payload["features"][str(args.layer)].float()

    rows = [
        json.loads(line)
        for line in Path(args.aligned_data_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != x.shape[0]:
        raise ValueError(
            f"Aligned data has {len(rows)} rows but features have {x.shape[0]}"
        )

    sub = torch.load(args.language_subspace_file, map_location="cpu")
    q_lang = sub["language_subspace_basis"].float()
    if int(sub["layer"]) != args.layer:
        raise ValueError("Language subspace layer mismatch")

    pair_to_indices = {}
    for i, row in enumerate(rows):
        pair_to_indices.setdefault(row["pair_id"], []).append(i)

    semantic_means = []
    for pair_id, idxs in pair_to_indices.items():
        if len(idxs) < 2:
            continue
        semantic_means.append(x[idxs].mean(dim=0))
    m = torch.stack(semantic_means)

    # Remove the linearly language-decodable component first.
    m = m - (m @ q_lang) @ q_lang.T
    m = m - m.mean(dim=0, keepdim=True)

    q = min(args.rank, m.shape[0], m.shape[1])
    _, _, v = torch.pca_lowrank(m, q=q, center=False)
    q_transfer = v[:, :q]
    # Numerical cleanup and explicit orthogonality to language basis.
    q_transfer = q_transfer - q_lang @ (q_lang.T @ q_transfer)
    q_transfer, _ = torch.linalg.qr(q_transfer, mode="reduced")
    q_transfer = q_transfer[:, :q]

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "layer": args.layer,
        "rank": int(q_transfer.shape[1]),
        "transferable_subspace_basis": q_transfer,
        "language_subspace_file": args.language_subspace_file,
        "features_file": args.features_file,
        "aligned_data_file": args.aligned_data_file,
        "definition": (
            "Top PCA directions of cross-language semantic means after removing "
            "the INLP language subspace."
        ),
    }, out)
    print(
        f"Saved transferable rank-{q_transfer.shape[1]} subspace at layer "
        f"{args.layer} to {out}"
    )


if __name__ == "__main__":
    main()
