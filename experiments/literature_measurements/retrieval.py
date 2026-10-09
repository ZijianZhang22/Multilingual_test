#!/usr/bin/env python3
"""Held-out, cross-lingual paired retrieval under language mean centering.

Inspired by Language Neutrality of Pre-trained Multilingual Representations.
Fitting uses separate multilingual pair features; evaluation uses NEVER-SEEN
aligned pair IDs. This measures same-item alignment, not task usage or memory.
"""
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F

from experiments.literature_measurements.core import save_csv


def read_bundle(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_metadata(fit, eva):
    if fit["pool"] != eva["pool"] or fit.get("source", "residual") != eva.get("source", "residual"):
        raise ValueError("Fit/eval source and pooling mismatch")
    for d in (fit, eva):
        if "pair_ids" not in d:
            raise ValueError("Use aligned features with pair_ids")
        if len(d["pair_ids"]) != len(d["languages"]):
            raise ValueError("Invalid pair_id lengths")
    a = set(str(v) for v in fit["pair_ids"] if v is not None)
    b = set(str(v) for v in eva["pair_ids"] if v is not None)
    if a & b:
        raise ValueError("Fit/eval multilingual pair IDs overlap")


def averaged_pairs(payload, layer):
    f = payload["features"][str(layer)].float()
    groups = defaultdict(list)
    for i, (lang, pid) in enumerate(zip(payload["languages"], payload["pair_ids"])):
        if pid is not None:
            groups[(lang, str(pid))].append(i)
    return {(lang, pid): f[idx].mean(0) for (lang, pid), idx in groups.items()}


def paired_retrieval(query, keys):
    q = F.normalize(query.float(), dim=-1)
    k = F.normalize(keys.float(), dim=-1)
    # query/key are ordered by the identical sorted pair_id list
    preds = (q @ k.T).argmax(-1)
    gold = torch.arange(len(query))
    return float(preds.eq(gold).float().mean())


def compute(fit, eva, layer, la, lb, setting):
    tr = averaged_pairs(fit, layer)
    te = averaged_pairs(eva, layer)
    tr_a = torch.stack([v for (lang,_),v in tr.items() if lang == la])
    tr_b = torch.stack([v for (lang,_),v in tr.items() if lang == lb])
    pair_ids = sorted({k[1] for k in te if k[0] == la} &
                      {k[1] for k in te if k[0] == lb})
    if len(pair_ids) < 3:
        return None
    a = torch.stack([te[la, pid] for pid in pair_ids])
    b = torch.stack([te[lb, pid] for pid in pair_ids])
    if setting == "raw":
        pass
    elif setting == "global_centered":
        cent = torch.cat([tr_a, tr_b]).mean(0)
        a, b = a - cent, b - cent
    elif setting == "language_centered":
        a, b = a - tr_a.mean(0), b - tr_b.mean(0)
    else:
        raise ValueError(setting)
    forward = paired_retrieval(a, b)
    reverse = paired_retrieval(b, a)
    return forward, reverse, len(pair_ids), len(tr_a), len(tr_b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor_fit", required=True)
    p.add_argument("--anchor_eval", required=True)
    p.add_argument("--adapted_fit", required=True)
    p.add_argument("--adapted_eval", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--languages", nargs="+", default=["en", "zh", "fr"])
    p.add_argument("--layers", default="all")
    args = p.parse_args()
    rows = []
    for name, fit_path, eva_path in (
        ("anchor", args.anchor_fit, args.anchor_eval),
        ("adapted", args.adapted_fit, args.adapted_eval)
    ):
        fit, eva = read_bundle(fit_path), read_bundle(eva_path)
        validate_metadata(fit, eva)
        layers = sorted(set(fit["features"]) & set(eva["features"]), key=int)
        if args.layers != "all":
            wanted = set(args.layers.replace(",", " ").split())
            if not wanted.issubset(layers):
                raise ValueError("Requested layer not available")
            layers = sorted(wanted, key=int)
        for layer_str in layers:
            layer = int(layer_str)
            for i, la in enumerate(args.languages):
                for lb in args.languages[i+1:]:
                    for setting in ("raw", "global_centered", "language_centered"):
                        got = compute(fit, eva, layer, la, lb, setting)
                        if got is None:
                            continue
                        ab, ba, n, tra, trb = got
                        rows.extend([
                            {"checkpoint": name, "layer": layer,
                             "source": eva.get("source", "residual"), "pool": eva["pool"],
                             "query_lang": la, "key_lang": lb,
                             "setting": setting, "recall_at_1": ab,
                             "n_eval_pairs": n, "n_fit_a": tra, "n_fit_b": trb},
                            {"checkpoint": name, "layer": layer,
                             "source": eva.get("source", "residual"), "pool": eva["pool"],
                             "query_lang": lb, "key_lang": la,
                             "setting": setting, "recall_at_1": ba,
                             "n_eval_pairs": n, "n_fit_a": trb, "n_fit_b": tra}
                        ])
            print(f"[{name}] L{layer} retrieval completed", flush=True)
    if not rows:
        raise ValueError("No valid paired retrieval comparisons")
    save_csv(args.out_csv, rows)
    print(f"Saved {len(rows)} rows: {args.out_csv}")


if __name__ == "__main__":
    main()
