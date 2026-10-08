#!/usr/bin/env python3
"""LayerMoE-style all-pairs cosine and cross-lingual aligned CKA.

Inputs are matching feature bundles from extract_attention.py (attention HSA),
or invariance/extract_hidden.py (residual mean/last states). These are
DIFFERENT measurement sites; the output includes the source and pooling.
"""
import argparse
from collections import defaultdict

import torch

from experiments.literature_measurements.core import (
    linear_cka, mean_pairwise_cosine, save_csv,
)


def indexes(payload, language, split):
    return torch.tensor([i for i, l in enumerate(payload["languages"])
                         if l == language and (split == "*" or
                             payload.get("splits", ["*"] * len(payload["languages"]))[i] == split)],
                        dtype=torch.long)


def paired_cka(payload, features, lang_a, lang_b, split):
    """Mean across tokens per aligned item; meaningful for same-item pairs."""
    pair_ids = payload.get("pair_ids")
    if not pair_ids:
        return None, 0
    group = defaultdict(lambda: defaultdict(list))
    for i, (lang, pair) in enumerate(zip(payload["languages"], pair_ids)):
        if pair is None or lang not in (lang_a, lang_b):
            continue
        if split != "*" and payload.get("splits", [split]*len(pair_ids))[i] != split:
            continue
        group[str(pair)][lang].append(i)
    keys = sorted(k for k, v in group.items() if lang_a in v and lang_b in v)
    if len(keys) < 3:
        return None, len(keys)
    x = torch.stack([features[group[k][lang_a]].mean(0) for k in keys])
    y = torch.stack([features[group[k][lang_b]].mean(0) for k in keys])
    return linear_cka(x, y), len(keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor_features", required=True)
    ap.add_argument("--adapted_features", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--languages", nargs="+", default=["en", "zh"])
    ap.add_argument("--split", default="probe_test")
    ap.add_argument("--layers", default="all")
    args = ap.parse_args()

    a = torch.load(args.anchor_features, map_location="cpu", weights_only=False)
    b = torch.load(args.adapted_features, map_location="cpu", weights_only=False)
    for key in ("languages", "example_ids", "splits", "pool"):
        if a.get(key) != b.get(key):
            raise ValueError(f"Anchor/adapted feature alignment mismatch: {key}")
    if a.get("source", "residual") != b.get("source", "residual"):
        raise ValueError("Different sites (attention vs residual); compare like with like")
    if a.get("pair_ids") != b.get("pair_ids"):
        raise ValueError("Anchor/adapted pair IDs mismatch")

    layers = sorted(int(i) for i in set(a["features"]) & set(b["features"]))
    if args.layers != "all":
        wanted = {int(s) for s in args.layers.replace(",", " ").split()}
        if not wanted.issubset(layers):
            raise ValueError(f"Missing layers: {wanted - set(layers)}")
        layers = sorted(wanted)

    result = []
    for layer in layers:
        fa, fb = a["features"][str(layer)].float(), b["features"][str(layer)].float()
        if fa.shape != fb.shape or fa.shape[0] != len(a["languages"]):
            raise ValueError(f"Invalid aligned tensors for layer {layer}")
        for lang_i, lang_a in enumerate(args.languages):
            for lang_b in args.languages[lang_i + 1:]:
                ia = indexes(a, lang_a, args.split)
                ib = indexes(a, lang_b, args.split)
                if len(ia) < 3 or len(ib) < 3:
                    raise ValueError(f"Missing >=3 examples at {layer}: {lang_a},{lang_b}")
                rows = {}
                for name, f in (("anchor", fa), ("adapted", fb)):
                    xa, xb = f[ia], f[ib]
                    joined_mean = torch.cat([xa, xb], dim=0).mean(dim=0)
                    centered = mean_pairwise_cosine(xa - joined_mean, xb - joined_mean)
                    raw = mean_pairwise_cosine(xa, xb)
                    cka, n_pairs = paired_cka(a, f, lang_a, lang_b, args.split)
                    rows[name] = {"raw": raw, "centered": centered,
                                  "cka": cka, "pairs": n_pairs}
                    result.append({
                        "checkpoint": name, "layer": layer,
                        "source": a.get("source", "residual"), "pool": a["pool"],
                        "language_a": lang_a, "language_b": lang_b, "split": args.split,
                        "examples_a": len(ia), "examples_b": len(ib),
                        "mean_pairwise_cosine": raw,
                        "global_centered_mean_pairwise_cosine": centered,
                        "aligned_pair_cka": "" if cka is None else cka,
                        "aligned_pairs": n_pairs,
                    })
                print(f"L{layer} {lang_a}/{lang_b}: "
                      f"raw {rows['anchor']['raw']:.4f}->{rows['adapted']['raw']:.4f} "
                      f"centered {rows['anchor']['centered']:.4f}->"
                      f"{rows['adapted']['centered']:.4f}", flush=True)
    save_csv(args.out_csv, result)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
