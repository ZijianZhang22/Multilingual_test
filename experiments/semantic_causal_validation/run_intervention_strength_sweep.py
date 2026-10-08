#!/usr/bin/env python3
"""Norm-controlled feature intervention sweep for semantic/task specificity.

Fits frozen linear probes on intact adapted hidden states, then perturbs only
held-out test representations. This avoids retraining a probe after ablation.

For each candidate basis:
  - natural component ablation sweep: x' = x - beta P_Q(x-center)
  - equal-energy sweep across same-rank candidates: all perturbations are scaled
    DOWN to the smallest natural projected energy in the comparison group.

Reports XNLI and language-ID accuracy/NLL changes. The equal-energy comparison
is the primary specificity test.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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


def load_bases(core_file, partition_file, random_draws):
    core = torch.load(core_file, map_location="cpu")
    bases = {}
    for name in ["transfer", "drift", "isr_cov", "isr_multiclass", "vicreg"]:
        if name in core["subspaces"]:
            bases[name] = orthonormalize(core["subspaces"][name].float())
    if partition_file:
        part = torch.load(partition_file, map_location="cpu")
        for name, q in part.get("derived_subspaces", {}).items():
            bases[name.replace("drift_isr_", "")] = orthonormalize(q.float())

    dim = int(core["hidden_dim"])
    for rank in [16, 32, 64]:
        for j in range(random_draws):
            bases[f"random{rank}_draw{j}"] = orthonormal_random(
                dim, rank, 9900 + rank * 100 + j
            )
    return core, bases


def train_probe(x, y, train_idx, n_classes, epochs, seed, device):
    return train_linear_classifier(
        x[train_idx].to(device),
        y[train_idx].to(device),
        n_classes,
        epochs,
        1e-2,
        1e-4,
        seed,
    )


def evaluate_probe(head, x, y):
    with torch.no_grad():
        logits = head(x)
        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        p = F.softmax(logits, dim=-1)
        gold = p[torch.arange(len(y), device=y.device), y].mean().item()
    return loss, acc, gold


def projected_component(x, center, q):
    xc = x.float() - center.float()
    return (xc @ q.float()) @ q.float().T


def rms_fraction(delta, reference):
    return float(
        delta.pow(2).mean().sqrt()
        / reference.float().pow(2).mean().sqrt().clamp_min(1e-20)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_file", required=True)
    ap.add_argument("--core_subspace_file", required=True)
    ap.add_argument("--partition_file", default=None)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--probe_epochs", type=int, default=120)
    ap.add_argument("--random_draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out_dir",
        default="mechanism_runs/semantic_causal_validation/intervention_sweep",
    )
    args = ap.parse_args()

    if any(b <= 0 or b > 1 for b in args.betas):
        raise ValueError("--betas must lie in (0,1].")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.features_file, map_location="cpu")
    x = payload["features"][str(args.layer)].float()
    labels = payload["labels"].long()
    languages = payload["languages"]
    splits = payload["splits"]

    core, bases = load_bases(
        args.core_subspace_file, args.partition_file, args.random_draws
    )
    if int(core["hidden_dim"]) != x.shape[1]:
        raise ValueError("Feature dimension and subspace dimension disagree.")
    center = core.get("center", x.mean(dim=0)).float()

    train_idx = torch.tensor(
        [i for i, s in enumerate(splits) if s == "probe_train"], dtype=torch.long
    )
    test_idx = torch.tensor(
        [i for i, s in enumerate(splits) if s == "probe_test"], dtype=torch.long
    )
    unique_langs = sorted(set(languages))
    lang_to_id = {l: i for i, l in enumerate(unique_langs)}
    lang_y = torch.tensor([lang_to_id[l] for l in languages], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_head = train_probe(
        x, labels, train_idx, int(labels.max()) + 1,
        args.probe_epochs, args.seed + 101, device
    )
    lang_head = train_probe(
        x, lang_y, train_idx, len(unique_langs),
        args.probe_epochs, args.seed + 202, device
    )

    xt = x[test_idx].to(device)
    yt = labels[test_idx].to(device)
    ylt = lang_y[test_idx].to(device)
    ct = center.to(device)

    base_task = evaluate_probe(task_head, xt, yt)
    base_lang = evaluate_probe(lang_head, xt, ylt)

    # Precompute natural test-set components and group same-rank candidates.
    components = {}
    rank_groups = defaultdict(list)
    natural_fracs = {}
    ref = xt - ct
    for name, q in bases.items():
        qd = q.to(device)
        comp = projected_component(xt, ct, qd)
        components[name] = comp
        natural_fracs[name] = rms_fraction(comp, ref)
        rank_groups[int(q.shape[1])].append(name)

    matched_scales = {}
    for rank, names in rank_groups.items():
        target = min(natural_fracs[n] for n in names)
        for n in names:
            matched_scales[n] = target / max(natural_fracs[n], 1e-20)

    rows = []
    for name, q in bases.items():
        comp = components[name]
        rank = int(q.shape[1])
        for mode, scale0 in [
            ("natural", 1.0),
            ("rank_energy_matched", matched_scales[name]),
        ]:
            for beta in args.betas:
                delta = beta * scale0 * comp
                altered = xt - delta
                task = evaluate_probe(task_head, altered, yt)
                lang = evaluate_probe(lang_head, altered, ylt)
                row = {
                    "subspace": name,
                    "rank": rank,
                    "pool": payload.get("pool", "unknown"),
                    "mode": mode,
                    "beta": beta,
                    "base_component_rms_fraction": natural_fracs[name],
                    "scale": scale0,
                    "actual_intervention_rms_fraction": rms_fraction(delta, ref),
                    "task_baseline_accuracy": base_task[1],
                    "task_accuracy": task[1],
                    "task_accuracy_change": task[1] - base_task[1],
                    "task_baseline_nll": base_task[0],
                    "task_nll": task[0],
                    "task_nll_increase": task[0] - base_task[0],
                    "task_gold_probability_drop": base_task[2] - task[2],
                    "language_baseline_accuracy": base_lang[1],
                    "language_accuracy": lang[1],
                    "language_accuracy_change": lang[1] - base_lang[1],
                    "language_baseline_nll": base_lang[0],
                    "language_nll": lang[0],
                    "language_nll_increase": lang[0] - base_lang[0],
                    "language_gold_probability_drop": base_lang[2] - lang[2],
                }
                rows.append(row)
                print(
                    f"{name:24s} {mode:19s} beta={beta:.2f} "
                    f"norm={row['actual_intervention_rms_fraction']:.4f} "
                    f"task_dNLL={row['task_nll_increase']:+.4f} "
                    f"lang_dNLL={row['language_nll_increase']:+.4f}",
                    flush=True,
                )

    write_csv(out / "intervention_strength_sweep.csv", rows)

    # Compact rank-32 primary table: top32/bottom32/random draws at equal energy.
    primary = [
        r for r in rows
        if r["mode"] == "rank_energy_matched"
        and r["rank"] == 32
        and (
            r["subspace"] in {"top32", "bottom32"}
            or r["subspace"].startswith("random32_draw")
        )
    ]
    write_csv(out / "rank32_primary_equal_energy.csv", primary)

    (out / "manifest.json").write_text(
        json.dumps(
            {
                **vars(args),
                "pool": payload.get("pool"),
                "primary_control": (
                    "Within each rank, every basis is scaled DOWN to the smallest "
                    "natural projected-component RMS fraction before the beta sweep. "
                    "No candidate is amplified."
                ),
                "warning": (
                    "This is a frozen-probe feature intervention. It complements, "
                    "but does not replace, end-to-end last-token model interventions."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {out / 'intervention_strength_sweep.csv'}")


if __name__ == "__main__":
    main()
