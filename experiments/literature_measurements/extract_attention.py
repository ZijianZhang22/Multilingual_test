#!/usr/bin/env python3
"""Extract post-self-attention outputs (LayerMoE-style HSA), or residual states.

Unlike invariance/extract_hidden.py this supports unpooled, masked TOKEN features.
Source JSONL can be produced by invariance/prepare_xnli.py or prepare_aligned_xnli.py.
"""
import argparse
from pathlib import Path

import torch

from experiments.literature_measurements.core import (
    get_layers, hidden_from_output, input_batch, layer_numbers, load_model, rows_from_jsonl
)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--layers", default="1,12,20,24")
    ap.add_argument("--source", choices=["attention", "residual"], default="attention")
    ap.add_argument("--pool", choices=["tokens", "mean", "last"], default="tokens")
    ap.add_argument("--split", default="probe_test", help="Use '*' to include all splits")
    ap.add_argument("--languages", nargs="+", default=["en", "zh", "fr", "de", "es"])
    ap.add_argument("--max_rows_per_language", type=int, default=200)
    ap.add_argument("--max_tokens_per_language", type=int, default=1500)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_bf16", action="store_true")
    a = ap.parse_args()

    data = [r for r in rows_from_jsonl(a.data_file, a.languages)
            if a.split == "*" or r.get("split") == a.split]
    if not data:
        raise ValueError("No rows matched selected split and languages")
    model, tok = load_model(a.checkpoint, a.device, bf16=not a.no_bf16)
    mods = get_layers(model)
    layer_ids = layer_numbers(a.layers, len(mods))
    buffers = {layer: [] for layer in layer_ids}
    captures = {}

    def make_hook(layer):
        def hook(_module, _inputs, output):
            captures[layer] = hidden_from_output(output).detach()
        return hook

    hooks = []
    for layer in layer_ids:
        target = mods[layer-1].self_attn if a.source == "attention" else mods[layer-1]
        hooks.append(target.register_forward_hook(make_hook(layer)))
    metadata = {"languages": [], "example_ids": [], "pair_ids": [], "splits": []}
    counters = {lang: 0 for lang in a.languages}
    try:
        for lang in a.languages:
            current = [r for r in data if r["language"] == lang][:a.max_rows_per_language]
            for pos in range(0, len(current), a.batch_size):
                if counters[lang] >= a.max_tokens_per_language:
                    break
                batch = current[pos:pos+a.batch_size]
                enc = input_batch(tok, [r["text"] for r in batch], a.max_length, a.device)
                model(**enc, use_cache=False)
                mask = enc["attention_mask"].bool()
                for j, row in enumerate(batch):
                    indices = mask[j].nonzero().flatten().tolist()
                    if a.pool == "tokens":
                        indices = indices[:max(0, a.max_tokens_per_language - counters[lang])]
                        if not indices:
                            continue
                        for layer in layer_ids:
                            buffers[layer].append(captures[layer][j, indices].float().cpu())
                        for k in indices:
                            metadata["languages"].append(lang)
                            metadata["example_ids"].append(f"{row.get('example_id', 'sample')}:{k}")
                            metadata["pair_ids"].append(row.get("pair_id"))
                            metadata["splits"].append(row.get("split", "unspecified"))
                        counters[lang] += len(indices)
                    else:
                        if not indices:
                            continue
                        for layer in layer_ids:
                            h = captures[layer][j, indices].float().cpu()
                            pooled = h.mean(0) if a.pool == "mean" else h[-1]
                            buffers[layer].append(pooled.unsqueeze(0))
                        metadata["languages"].append(lang)
                        metadata["example_ids"].append(str(row.get("example_id", f"{lang}:{pos+j}")))
                        metadata["pair_ids"].append(row.get("pair_id"))
                        metadata["splits"].append(row.get("split", "unspecified"))
                        counters[lang] += 1
                captures.clear()
    finally:
        for hook in hooks:
            hook.remove()
    if not metadata["example_ids"]:
        raise ValueError("No tokens extracted")
    payload = {
        "checkpoint": a.checkpoint, "pool": a.pool, "source": a.source,
        "layers": layer_ids, "features": {str(k): torch.cat(v, dim=0)
                                       for k, v in buffers.items()},
        **metadata
    }
    out = Path(a.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"Saved {len(metadata['example_ids'])} {a.pool} representations to {out}")
    print(f"Counts by language: {counters}")


if __name__ == "__main__":
    main()
