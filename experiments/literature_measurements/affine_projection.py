#!/usr/bin/env python3
"""Chang et al.-inspired multilingual affine-subspace LM intervention.

Measure SAME / CROSS / CROSS_RECENTER projections on identical held-out text,
separately for anchor and adapted Qwen checkpoints. This is a decoder-only
adaptation of the multilingual masked-LM experiment, not an exact replication.
"""
import argparse
import json
from pathlib import Path

import torch

from experiments.literature_measurements.core import (
    affine_project, ensure_disjoint, fit_affine, get_layers, hidden_from_output,
    input_batch, layer_numbers, load_model, rows_from_jsonl, save_csv, with_hidden,
)


@torch.no_grad()
def fit_token_features(model, tok, rows, langs, layers, device, args):
    """Use SAME token positions that interventions will modify: all tokens."""
    modules = get_layers(model)
    vectors = {lang: {layer: [] for layer in layers} for lang in langs}
    captured = {}

    def make_hook(layer):
        def hook(_module, _input, output):
            captured[layer] = hidden_from_output(output).detach()
        return hook

    hooks = [modules[layer - 1].register_forward_hook(make_hook(layer)) for layer in layers]
    try:
        for lang in langs:
            subset = [r for r in rows if r["language"] == lang][:args.fit_examples_per_lang]
            if not subset:
                raise ValueError(f"No fitting samples for {lang}")
            counts = {layer: 0 for layer in layers}
            for start in range(0, len(subset), args.batch_size):
                batch = subset[start:start + args.batch_size]
                enc = input_batch(tok, [r["text"] for r in batch], args.max_length, device)
                model(**enc, use_cache=False)
                mask = enc["attention_mask"].bool()
                for layer in layers:
                    if counts[layer] >= args.max_fit_tokens_per_lang:
                        continue
                    h = captured[layer][mask].float().cpu()
                    n = min(args.max_fit_tokens_per_lang - counts[layer], h.shape[0])
                    if n > 0:
                        vectors[lang][layer].append(h[:n])
                        counts[layer] += n
                captured.clear()
                if all(v >= args.max_fit_tokens_per_lang for v in counts.values()):
                    break
    finally:
        for h in hooks:
            h.remove()

    return {lang: {layer: torch.cat(vectors[lang][layer], dim=0)
                   for layer in layers} for lang in langs}


@torch.no_grad()
def loss_for_rows(model, tok, rows, device, batch_size, max_length, *, layer=None, projection=None):
    hooks = []
    if layer is not None and projection is not None:
        mu_in, mu_target, q, mode = projection

        def hook(_module, _inputs, output):
            h = hidden_from_output(output)
            new = affine_project(h, mu_in, mu_target, q, mode=mode)
            return with_hidden(output, new)

        hooks.append(get_layers(model)[layer-1].register_forward_hook(hook))
    total_nll = 0.0
    total_tokens = 0
    try:
        for i in range(0, len(rows), batch_size):
            sample = rows[i:i+batch_size]
            enc = input_batch(tok, [r["text"] for r in sample], max_length, device, labels=True)
            count = int(enc["attention_mask"][:, 1:].sum().item())
            if count == 0:
                continue
            out = model(**enc, use_cache=False)
            total_nll += float(out.loss.detach()) * count
            total_tokens += count
    finally:
        for h in hooks:
            h.remove()
    if total_tokens == 0:
        raise ValueError("Evaluation had zero target tokens")
    return total_nll / total_tokens, total_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor_checkpoint", required=True)
    p.add_argument("--adapted_checkpoint", required=True)
    p.add_argument("--fit_jsonl", required=True, help="Fit-only rows; disjoint from eval rows")
    p.add_argument("--eval_jsonl", required=True, help="Held-out evaluation rows")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--languages", nargs=2, default=["en", "zh"])
    p.add_argument("--layers", default="12,20,24", help="1-based decoder layers, or all")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--fit_examples_per_lang", type=int, default=256)
    p.add_argument("--eval_examples_per_lang", type=int, default=100)
    p.add_argument("--max_fit_tokens_per_lang", type=int, default=8192)
    p.add_argument("--variance_target", type=float, default=0.90)
    p.add_argument("--max_rank", type=int, default=256)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()

    if not 0 < args.variance_target <= 1:
        p.error("--variance_target must be in (0,1]")
    if args.max_rank < 1:
        p.error("--max_rank must be positive")
    langs = args.languages
    fit_rows = rows_from_jsonl(args.fit_jsonl, langs)
    eval_rows = rows_from_jsonl(args.eval_jsonl, langs)
    ensure_disjoint(fit_rows, eval_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    measurements = []
    basis_meta = []
    # Load only one checkpoint at a time, not two 3B models simultaneously.
    for name, checkpoint in (("anchor", args.anchor_checkpoint),
                             ("adapted", args.adapted_checkpoint)):
        print(f"[{name}] checkpoint: {checkpoint}", flush=True)
        model, tok = load_model(checkpoint, args.device, bf16=not args.no_bf16)
        layer_ids = layer_numbers(args.layers, len(get_layers(model)))
        features = fit_token_features(model, tok, fit_rows, langs, layer_ids, args.device, args)
        bases = {}
        for lang in langs:
            for layer in layer_ids:
                mu, q, meta = fit_affine(features[lang][layer],
                                        target_variance=args.variance_target,
                                        max_rank=args.max_rank,
                                        seed=args.seed + layer)
                bases[(lang, layer)] = (mu, q)
                basis_meta.append({"checkpoint": name, "language": lang, "layer": layer, **meta})
                print(f"[{name}] {lang} L{layer} rank={meta['rank']} "
                      f"variance={meta['explained_variance']:.3f} "
                      f"target_reached={meta['target_reached']}", flush=True)
        torch.save({"bases": bases, "metadata": [r for r in basis_meta
                      if r["checkpoint"] == name], "fit_jsonl": args.fit_jsonl},
                   out_dir / f"{name}_affine_bases.pt")

        for lang in langs:
            subset = [r for r in eval_rows if r["language"] == lang][:args.eval_examples_per_lang]
            if not subset:
                raise ValueError(f"No evaluation examples for {lang}")
            base, tokens = loss_for_rows(model, tok, subset, args.device,
                                         args.batch_size, args.max_length)
            for layer in layer_ids:
                own_mu, own_q = bases[(lang, layer)]
                other_lang = next(l for l in langs if l != lang)
                other_mu, other_q = bases[(other_lang, layer)]
                for mode in ("same", "cross", "cross_recenter"):
                    mu_q = (own_mu, own_mu, own_q, "same") if mode == "same" else (
                        own_mu, other_mu, other_q, mode
                    )
                    val, _ = loss_for_rows(model, tok, subset, args.device,
                                           args.batch_size, args.max_length,
                                           layer=layer, projection=mu_q)
                    measurements.append({
                        "checkpoint": name, "layer": layer, "language": lang,
                        "projection": mode, "basis_language": lang if mode == "same" else other_lang,
                        "baseline_nll": base, "intervention_nll": val,
                        "delta_nll": val-base,
                        "ppl_ratio": __import__("math").exp(min(30, val-base)),
                        "eval_tokens": tokens, "rank": mu_q[2].shape[1],
                        "target_reached": next(r["target_reached"] for r in basis_meta
                                               if r["checkpoint"] == name and r["layer"] == layer
                                               and r["language"] ==
                                               (lang if mode == "same" else other_lang)),
                    })
                print(f"[{name}] evaluated {lang} layer {layer}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_csv(out_dir / "affine_projection_results.csv", measurements)
    save_csv(out_dir / "affine_basis_metadata.csv", basis_meta)
    (out_dir / "protocol.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Saved {len(measurements)} intervention rows in {out_dir}")


if __name__ == "__main__":
    main()
