#!/usr/bin/env python3
"""Semantic-specificity validation for multilingual adaptation subspaces.

This script asks whether a candidate subspace is more strongly associated with
semantic identity than with language identity.

It combines three complementary tests:
  1) probe selectivity on XNLI: task label vs language identity,
  2) cross-lingual aligned retrieval: does a view retrieve the same semantic
     item in another language,
  3) semantic separation: same-item cross-language cosine vs unrelated controls.

The script can evaluate core subspaces and Step-7 Drift/ISR top/bottom
partitions. It is descriptive/diagnostic: it does not by itself prove that a
subspace "is semantics".
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from invariance.benchmark_representation_methods import train_linear_classifier
from experiments.retention_subspace_mechanism.subspace_extractors import (
    orthonormal_random,
    orthonormalize,
)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def probe_accuracy(z, y, train_idx, test_idx, n_classes, epochs, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = z.float().to(device)
    y = y.long().to(device)
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)
    head = train_linear_classifier(
        z[train_idx], y[train_idx], n_classes, epochs, 1e-2, 1e-4, seed
    )
    with torch.no_grad():
        pred = head(z[test_idx]).argmax(dim=-1)
        return float((pred == y[test_idx]).float().mean().cpu())


def project(x, q, center=None):
    x = x.float()
    if center is not None:
        x = x - center.float()
    return x @ q.float()


def cosine_rows(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def aligned_retrieval(z, rows):
    """Cross-language nearest-neighbor retrieval of semantic pair id."""
    langs = sorted(set(r["language"] for r in rows))
    by_lang = {}
    for lang in langs:
        idx = [i for i, r in enumerate(rows) if r["language"] == lang]
        by_lang[lang] = torch.tensor(idx, dtype=torch.long)

    z = F.normalize(z.float(), dim=-1)
    correct = 0
    total = 0
    per_pair = []

    for src in langs:
        src_idx = by_lang[src]
        for tgt in langs:
            if src == tgt:
                continue
            tgt_idx = by_lang[tgt]
            sims = z[src_idx] @ z[tgt_idx].T
            nn = sims.argmax(dim=1)
            c = 0
            for qi, ni in enumerate(nn.tolist()):
                src_row = rows[int(src_idx[qi])]
                tgt_row = rows[int(tgt_idx[ni])]
                ok = src_row["pair_id"] == tgt_row["pair_id"]
                correct += int(ok)
                c += int(ok)
                total += 1
            per_pair.append({
                "src_language": src,
                "tgt_language": tgt,
                "accuracy": c / max(len(src_idx), 1),
                "n": int(len(src_idx)),
            })
    return correct / max(total, 1), per_pair


def semantic_cosine_stats(z, rows, seed=0, max_pairs=5000):
    """Compare cross-language same-item similarity with semantic controls."""
    g = torch.Generator().manual_seed(seed)
    by_pair = {}
    for i, r in enumerate(rows):
        by_pair.setdefault(r["pair_id"], []).append(i)

    positive = []
    same_label_diff_item = []
    diff_label = []
    all_idx = torch.arange(len(rows))
    zn = F.normalize(z.float(), dim=-1)

    # Positive: same semantic pair, different languages.
    for idxs in by_pair.values():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                if rows[a]["language"] != rows[b]["language"]:
                    positive.append(float((zn[a] * zn[b]).sum()))

    # Matched random controls with different pair ids.
    n_draw = min(max_pairs, max(len(positive), 1))
    for _ in range(n_draw * 5):
        if len(same_label_diff_item) >= n_draw and len(diff_label) >= n_draw:
            break
        a = int(all_idx[torch.randint(len(rows), (1,), generator=g)])
        b = int(all_idx[torch.randint(len(rows), (1,), generator=g)])
        if a == b or rows[a]["pair_id"] == rows[b]["pair_id"]:
            continue
        sim = float((zn[a] * zn[b]).sum())
        if rows[a]["label"] == rows[b]["label"]:
            if len(same_label_diff_item) < n_draw:
                same_label_diff_item.append(sim)
        else:
            if len(diff_label) < n_draw:
                diff_label.append(sim)

    def m(x):
        return float(sum(x) / max(len(x), 1))

    def s(x):
        if len(x) < 2:
            return 0.0
        mu = m(x)
        return math.sqrt(sum((v - mu) ** 2 for v in x) / (len(x) - 1))

    return {
        "same_item_crosslang_cosine_mean": m(positive),
        "same_item_crosslang_cosine_std": s(positive),
        "same_label_diff_item_cosine_mean": m(same_label_diff_item),
        "diff_label_cosine_mean": m(diff_label),
        "semantic_pair_margin_vs_same_label": m(positive) - m(same_label_diff_item),
        "semantic_pair_margin_vs_diff_label": m(positive) - m(diff_label),
        "n_positive_pairs": len(positive),
        "n_control_pairs": min(len(same_label_diff_item), len(diff_label)),
    }


def load_bases(core_file, partition_file, random_draws):
    payload = torch.load(core_file, map_location="cpu")
    bases = {}
    for name in ["transfer", "drift", "isr_cov", "isr_multiclass", "vicreg"]:
        if name in payload["subspaces"]:
            bases[name] = orthonormalize(payload["subspaces"][name].float())

    if partition_file:
        part = torch.load(partition_file, map_location="cpu")
        for name, q in part.get("derived_subspaces", {}).items():
            bases[name.replace("drift_isr_", "")] = orthonormalize(q.float())

    dim = int(payload["hidden_dim"])
    for rank in [16, 32, 64]:
        for j in range(random_draws):
            bases[f"random{rank}_draw{j}"] = orthonormal_random(
                dim, rank, 8800 + rank * 100 + j
            )
    return payload, bases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_features", required=True)
    ap.add_argument("--aligned_features", required=True)
    ap.add_argument("--probe_data", required=True)
    ap.add_argument("--aligned_data", required=True)
    ap.add_argument("--core_subspace_file", required=True)
    ap.add_argument("--partition_file", default=None)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--probe_epochs", type=int, default=120)
    ap.add_argument("--random_draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out_dir",
        default="mechanism_runs/semantic_causal_validation/semantic_specificity",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    probe_payload = torch.load(args.probe_features, map_location="cpu")
    aligned_payload = torch.load(args.aligned_features, map_location="cpu")
    probe_rows = read_jsonl(args.probe_data)
    aligned_rows = read_jsonl(args.aligned_data)

    if probe_payload.get("pool") != aligned_payload.get("pool"):
        raise ValueError("Probe and aligned features must use the same pooling mode.")

    x_probe = probe_payload["features"][str(args.layer)].float()
    x_aligned = aligned_payload["features"][str(args.layer)].float()
    if len(probe_rows) != x_probe.shape[0] or len(aligned_rows) != x_aligned.shape[0]:
        raise ValueError("Feature/data length mismatch.")

    core, bases = load_bases(
        args.core_subspace_file, args.partition_file, args.random_draws
    )
    center = core.get("center", x_probe.mean(dim=0)).float()

    langs = probe_payload["languages"]
    unique_langs = sorted(set(langs))
    lang_to_id = {l: i for i, l in enumerate(unique_langs)}
    lang_y = torch.tensor([lang_to_id[l] for l in langs], dtype=torch.long)
    task_y = probe_payload["labels"].long()
    train_idx = torch.tensor(
        [i for i, s in enumerate(probe_payload["splits"]) if s == "probe_train"],
        dtype=torch.long,
    )
    test_idx = torch.tensor(
        [i for i, s in enumerate(probe_payload["splits"]) if s == "probe_test"],
        dtype=torch.long,
    )

    rows = []
    retrieval_rows = []
    for bi, (name, q) in enumerate(bases.items()):
        z_probe = project(x_probe, q, center)
        z_aligned = project(x_aligned, q, center)

        task_acc = probe_accuracy(
            z_probe,
            task_y,
            train_idx,
            test_idx,
            int(task_y.max()) + 1,
            args.probe_epochs,
            args.seed + 1000 + bi,
        )
        lang_acc = probe_accuracy(
            z_probe,
            lang_y,
            train_idx,
            test_idx,
            len(unique_langs),
            args.probe_epochs,
            args.seed + 2000 + bi,
        )
        retrieval, pair_rows = aligned_retrieval(z_aligned, aligned_rows)
        cos = semantic_cosine_stats(
            z_aligned, aligned_rows, seed=args.seed + 3000 + bi
        )

        row = {
            "subspace": name,
            "rank": int(q.shape[1]),
            "pool": probe_payload.get("pool", "unknown"),
            "task_probe_accuracy": task_acc,
            "language_probe_accuracy": lang_acc,
            "language_chance_accuracy": 1.0 / len(unique_langs),
            "crosslingual_same_item_retrieval_accuracy": retrieval,
            **cos,
        }
        rows.append(row)

        for pr in pair_rows:
            retrieval_rows.append({"subspace": name, **pr})

        print(
            f"{name:24s} rank={q.shape[1]:3d} "
            f"task={task_acc:.3f} lang={lang_acc:.3f} "
            f"retrieval={retrieval:.3f} "
            f"pair_margin={cos['semantic_pair_margin_vs_same_label']:+.3f}",
            flush=True,
        )

    write_csv(out / "semantic_specificity_summary.csv", rows)
    write_csv(out / "retrieval_by_language_pair.csv", retrieval_rows)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                **vars(args),
                "pool": probe_payload.get("pool"),
                "interpretation": {
                    "task_probe_accuracy": "higher means more XNLI-label information is linearly available",
                    "language_probe_accuracy": "closer to chance means less language identity is linearly available",
                    "crosslingual_same_item_retrieval_accuracy": "higher means semantic-item identity is preserved across languages",
                    "semantic_pair_margin_vs_same_label": "positive means same semantic item is closer than unrelated items with the same NLI label",
                },
                "warning": "These are representation diagnostics, not sufficient alone for semantic causal identification.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {out / 'semantic_specificity_summary.csv'}")


if __name__ == "__main__":
    main()
