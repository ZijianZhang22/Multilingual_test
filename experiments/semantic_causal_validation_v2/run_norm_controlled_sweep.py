#!/usr/bin/env python3
"""Exact-norm causal ablation sweep on NLI and language-ID behavior."""

import argparse
import csv
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.random_layer_freeze_pilot.run_pilot import load_model  # noqa: E402
from experiments.semantic_causal_validation_v2.common import (  # noqa: E402
    build_last_token_ablation_hook, choice_token_ids, forward_choice,
    language_prompt, load_xnli_rows, nli_prompt, parse_basis_specs,
    save_json, seed_all,
)


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["task"], r["subspace"], r["norm_fraction"]), []).append(r)
    out = []
    for (task, subspace, frac), xs in sorted(groups.items()):
        n = len(xs)
        out.append({
            "task": task,
            "subspace": subspace,
            "norm_fraction": frac,
            "n": n,
            "accuracy": sum(x["accuracy"] for x in xs) / n,
            "mean_gold_probability": sum(x["gold_probability"] for x in xs) / n,
            "mean_gold_nll": sum(x["gold_nll"] for x in xs) / n,
            "mean_gold_probability_drop_vs_baseline": sum(x["gold_probability_drop_vs_baseline"] for x in xs) / n,
            "mean_nll_increase_vs_baseline": sum(x["nll_increase_vs_baseline"] for x in xs) / n,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--basis_file", default=None)
    ap.add_argument("--basis", nargs="+", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--languages", nargs="+", default=["en", "zh", "fr"])
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--n_per_lang", type=int, default=60)
    ap.add_argument("--norm_fractions", nargs="+", type=float, default=[0.03, 0.10, 0.20, 0.40, 0.60])
    ap.add_argument("--tasks", nargs="+", choices=["nli", "language_id"], default=["nli", "language_id"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_dir", default="mechanism_runs/semantic_causal_validation_v2/norm_controlled_sweep")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    seed_all(args.seed)
    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device, use_bf16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bases = parse_basis_specs(args.basis, args.basis_file)
    rows_data = load_xnli_rows(args.languages, args.split, args.n_per_lang, args.seed)
    nli_ids = choice_token_ids(tokenizer, ["A", "B", "C"])
    lang_letters = [chr(ord("A") + i) for i in range(len(args.languages))]
    lang_ids = choice_token_ids(tokenizer, lang_letters)
    lang_to_idx = {lang: i for i, lang in enumerate(args.languages)}

    baseline = {}
    for task in args.tasks:
        for row in rows_data:
            prompt = nli_prompt(row) if task == "nli" else language_prompt(row, args.languages)
            ids = nli_ids if task == "nli" else lang_ids
            gold = row["label"] if task == "nli" else lang_to_idx[row["language"]]
            probs = forward_choice(model, tokenizer, prompt, ids, device=device, use_bf16=use_bf16)
            baseline[(task, row["example_id"])] = {
                "gold_probability": float(probs[gold]),
                "gold_nll": -float(torch.log(probs[gold].clamp_min(1e-30))),
                "accuracy": float(int(probs.argmax()) == gold),
            }

    result_rows = []
    for task in args.tasks:
        ids = nli_ids if task == "nli" else lang_ids
        for name, item in bases.items():
            q, center = item["q"], item["center"]
            for frac in args.norm_fractions:
                print(f"[{task}] {name} norm_fraction={frac:g}", flush=True)
                register = build_last_token_ablation_hook(model, args.layer, q, center, frac)
                for row in rows_data:
                    prompt = nli_prompt(row) if task == "nli" else language_prompt(row, args.languages)
                    gold = row["label"] if task == "nli" else lang_to_idx[row["language"]]
                    probs = forward_choice(model, tokenizer, prompt, ids, device=device, use_bf16=use_bf16, hook=register)
                    p = float(probs[gold])
                    nll = -float(torch.log(probs[gold].clamp_min(1e-30)))
                    b = baseline[(task, row["example_id"])]
                    result_rows.append({
                        "task": task,
                        "example_id": row["example_id"],
                        "language": row["language"],
                        "label": gold,
                        "subspace": name,
                        "rank": int(q.shape[1]),
                        "norm_fraction": frac,
                        "accuracy": float(int(probs.argmax()) == gold),
                        "gold_probability": p,
                        "gold_nll": nll,
                        "baseline_accuracy": b["accuracy"],
                        "baseline_gold_probability": b["gold_probability"],
                        "baseline_gold_nll": b["gold_nll"],
                        "gold_probability_drop_vs_baseline": b["gold_probability"] - p,
                        "nll_increase_vs_baseline": nll - b["gold_nll"],
                    })

    example_file = out / "example_level.csv"
    with example_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        w.writeheader()
        w.writerows(result_rows)

    summary = summarize(result_rows)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    save_json(out / "protocol.json", {
        **vars(args),
        "intervention": "last-token projected ablation with exact per-example perturbation norm matching",
        "feature_mismatch_note": "Use semantic_selectivity for a pooling-aligned mean-state intervention.",
        "basis_metadata": {k: v["meta"] for k, v in bases.items()},
    })
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
