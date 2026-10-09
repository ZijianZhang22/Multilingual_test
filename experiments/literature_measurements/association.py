#!/usr/bin/env python3
"""Join layerwise sharedness with full-layer patching, and report associations.

Correlations are descriptive, not causal; layer samples are strongly dependent.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from experiments.literature_measurements.core import save_csv


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_rank(x):
    """Midranks for ties (Spearman)."""
    indexed = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    at = 0
    while at < len(x):
        end = at + 1
        while end < len(x) and x[indexed[end]] == x[indexed[at]]:
            end += 1
        rank = 0.5 * (at + 1 + end)
        for j in range(at, end):
            ranks[indexed[j]] = rank
        at = end
    return ranks


def pearson(x, y):
    if len(x) < 4:
        return None
    xm = sum(x)/len(x)
    ym = sum(y)/len(y)
    xx = sum((a-xm)**2 for a in x)
    yy = sum((b-ym)**2 for b in y)
    if xx * yy <= 0:
        return None
    return sum((a-xm)*(b-ym) for a, b in zip(x,y)) / math.sqrt(xx*yy)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--similarity_csv", required=True)
    p.add_argument("--patch_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--language_a", default="en")
    p.add_argument("--language_b", default="zh")
    p.add_argument("--old_language", default="en")
    a = p.parse_args()
    sim = {}
    for r in read_csv(a.similarity_csv):
        if {r["language_a"], r["language_b"]} != {a.language_a, a.language_b}:
            continue
        sim.setdefault(int(r["layer"]), {})[r["checkpoint"]] = r
    patch = {int(r["layer"]): r for r in read_csv(a.patch_csv)
             if r["language"] == a.old_language}
    rows = []
    for layer in sorted(set(sim) & set(patch)):
        pair = sim[layer]
        if "anchor" not in pair or "adapted" not in pair:
            continue
        ar, br = pair["anchor"], pair["adapted"]
        pr = patch[layer]
        delta_raw = float(br["mean_pairwise_cosine"]) - float(ar["mean_pairwise_cosine"])
        delta_center = float(br["global_centered_mean_pairwise_cosine"]) - float(
            ar["global_centered_mean_pairwise_cosine"])
        rows.append({
            "layer": layer, "similarity_source": br["source"],
            "pool": br["pool"], "language_pair": a.language_a + "-" + a.language_b,
            "similarity_anchor": float(ar["mean_pairwise_cosine"]),
            "similarity_adapted": float(br["mean_pairwise_cosine"]),
            "similarity_change": delta_raw,
            "centered_similarity_change": delta_center,
            "old_patch_nll_improvement": -float(pr["patch_delta_loss"]),
            "old_gap_recovery": pr["patch_gap_recovery"],
            "weight_restore_nll_improvement": (
                "" if not pr["weight_restore_loss"] else
                float(pr["adapted_loss"]) - float(pr["weight_restore_loss"])),
        })
    if not rows:
        raise ValueError("No jointly measured layers")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "layer_sharedness_patch_join.csv", rows)
    report = {"n_layers": len(rows), "interpretation":
        "Exploratory layer-level association; not an independent-sample causal test"}
    y = [r["old_patch_nll_improvement"] for r in rows]
    for name in ("similarity_anchor", "similarity_adapted", "similarity_change",
                 "centered_similarity_change"):
        x = [r[name] for r in rows]
        report[name] = {
            "pearson": pearson(x, y),
            "spearman": pearson(mean_rank(x), mean_rank(y))
        }
    (out / "layer_sharedness_correlations.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {len(rows)} layers, {out}")


if __name__ == "__main__":
    main()
