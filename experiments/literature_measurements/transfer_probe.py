#!/usr/bin/env python3
"""Matched-rank cross-lingual task probes for existing subspaces.

Evaluate monolingual, English direct transfer, leave-one-language-out,
and joint-language XNLI classification. Fit probes only on probe_train;
evaluate only on probe_test. For valid causality claims, supplied external
bases must also have been fitted without viewing probe_test features.
"""
import argparse

import torch
import torch.nn.functional as F

from experiments.literature_measurements.core import orthonormal, save_csv


def fit_ridge(x, labels, n_classes, ridge):
    x = x.float()
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-4)
    x = (x - mean) / std
    x = torch.cat([x, torch.ones(len(x), 1)], dim=1)
    y = F.one_hot(labels.long(), num_classes=n_classes).float()
    eye = torch.eye(x.shape[1])
    eye[-1, -1] = 0  # unregularized intercept
    w = torch.linalg.solve(x.T @ x + ridge * eye + 1e-6 * torch.eye(x.shape[1]), x.T @ y)
    return mean, std, w


def accuracy(model, x, labels):
    mean, std, w = model
    x = (x.float() - mean) / std
    x = torch.cat([x, torch.ones(len(x), 1)], dim=1)
    return float((x @ w).argmax(-1).eq(labels.long()).float().mean())


def load_basis_bundle(path, layer):
    if not path:
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "layer" in payload and int(payload["layer"]) != layer:
        raise ValueError(f"External basis layer {payload['layer']} != {layer}")
    subspaces = payload.get("subspaces", payload.get("anchor_subspaces"))
    if subspaces is None:
        raise ValueError("Bundle needs 'subspaces' or 'anchor_subspaces' mapping")
    return {str(k): orthonormal(v) for k, v in subspaces.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor_features", required=True)
    p.add_argument("--adapted_features", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--layers", default="all")
    p.add_argument("--languages", nargs="+", default=["en", "zh", "fr", "de", "es"])
    p.add_argument("--source_language", default="en")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--basis_file", help="Optional trusted fit-only external basis, one layer")
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    if a.rank < 1 or a.ridge < 0:
        p.error("rank must be positive and ridge nonnegative")

    anchor = torch.load(a.anchor_features, map_location="cpu", weights_only=False)
    adapted = torch.load(a.adapted_features, map_location="cpu", weights_only=False)
    for k in ("example_ids", "splits", "languages"):
        if anchor[k] != adapted[k]:
            raise ValueError(f"Misaligned anchor/adapted rows: {k}")
    if not torch.equal(anchor["labels"], adapted["labels"]):
        raise ValueError("Label mismatch")
    labels = anchor["labels"].long()
    if labels.ndim != 1 or (labels < 0).any():
        raise ValueError("Labels must be nonnegative class IDs")
    n_classes = int(labels.max()) + 1
    langs = anchor["languages"]
    splits = anchor["splits"]
    all_layers = sorted(set(anchor["features"]) & set(adapted["features"]), key=int)
    layers = all_layers if a.layers == "all" else a.layers.replace(",", " ").split()
    if not set(layers).issubset(set(all_layers)):
        raise ValueError("Feature layer missing")
    rows = []

    for lstr in layers:
        layer = int(lstr)
        x_anchor = anchor["features"][lstr].float()
        x_adapted = adapted["features"][lstr].float()
        if x_anchor.shape != x_adapted.shape:
            raise ValueError("Feature shape mismatch")
        idx = torch.tensor([i for i, s in enumerate(splits)
                            if s == "probe_train" and langs[i] in a.languages])
        if len(idx) < 3:
            raise ValueError("No fitting split present")
        n = min(a.rank, x_anchor.shape[1], len(idx) - 1)
        torch.manual_seed(a.seed + layer)
        xc = x_anchor[idx] - x_anchor[idx].mean(0)
        _, _, v = torch.pca_lowrank(xc, q=n, center=False, niter=3)
        q_anchor = orthonormal(v[:, :n])

        delta = x_adapted[idx] - x_anchor[idx]
        dc = delta - delta.mean(0)
        _, _, v = torch.pca_lowrank(dc, q=n, center=False, niter=3)
        q_drift = orthonormal(v[:, :n])
        q_random = orthonormal(torch.randn(x_anchor.shape[1], n,
                               generator=torch.Generator().manual_seed(a.seed + 1000 + layer)))
        spaces = {"anchor_pca": q_anchor, "drift_train": q_drift,
                  "random": q_random}
        spaces.update(load_basis_bundle(a.basis_file, layer))

        for checkpoint, x in (("anchor", x_anchor), ("adapted", x_adapted)):
            for space_name, q in spaces.items():
                if q.shape[0] != x.shape[1]:
                    raise ValueError(f"Basis {space_name} dimension mismatch")
                projected = x @ q
                for task in ("monolingual", "direct", "loo", "joint"):
                    for target in a.languages:
                        if task == "direct" and target == a.source_language:
                            continue
                        if task == "direct":
                            train_langs = {a.source_language}
                        elif task == "monolingual":
                            train_langs = {target}
                        elif task == "loo":
                            train_langs = set(a.languages) - {target}
                        else:
                            train_langs = set(a.languages)
                        tr = torch.tensor([i for i in range(len(langs))
                                           if splits[i] == "probe_train" and langs[i] in train_langs])
                        te = torch.tensor([i for i in range(len(langs))
                                           if splits[i] == "probe_test" and langs[i] == target])
                        if len(tr) < 3 or len(te) < 3 or len(torch.unique(labels[tr])) < 2:
                            continue
                        model = fit_ridge(projected[tr], labels[tr], n_classes, a.ridge)
                        rows.append({
                            "checkpoint": checkpoint, "layer": layer,
                            "basis": space_name, "rank": int(q.shape[1]),
                            "task": task, "source_langs": ",".join(sorted(train_langs)),
                            "target_language": target, "n_train": len(tr), "n_test": len(te),
                            "accuracy": accuracy(model, projected[te], labels[te]),
                            "basis_fit_protocol": "train-only" if space_name in
                                ("anchor_pca", "drift_train", "random") else "external-unverified",
                        })
            print(f"[{checkpoint}] L{layer}: {len(rows)} total task-probe rows", flush=True)
    save_csv(a.out_csv, rows)
    print(f"Saved {len(rows)} results to {a.out_csv}")


if __name__ == "__main__":
    main()
