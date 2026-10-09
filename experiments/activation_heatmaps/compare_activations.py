#!/usr/bin/env python3
"""Fixed-input before/after activation comparison; never generates text."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def records(path, limit):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    rows = rows[:limit] if limit else rows
    if len(rows) < 2 or any(not isinstance(r.get('text'), str) or not r['text'].strip() for r in rows):
        raise ValueError('Need at least two JSONL rows with nonempty text.')
    ids = [str(r.get('id', i)) for i, r in enumerate(rows)]
    if len(set(ids)) != len(ids):
        raise ValueError('Sample IDs must be unique.')
    for r, sid in zip(rows, ids):
        r['id'] = sid
    return rows


def extract(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = records(args.inputs, args.limit)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map=args.device,
        attn_implementation='eager')
    model.eval()
    # Hook decoder block outputs directly: avoids the final norm ambiguity
    # present in some architectures' output_hidden_states tuples.
    blocks = {int(m.group(1)): (name, module) for name, module in model.named_modules()
              if (m := re.fullmatch(r'.*layers\.(\d+)', name))}
    if not blocks or sorted(blocks) != list(range(len(blocks))):
        raise ValueError('This pilot supports Qwen/Llama/Gemma-style layers.N decoders.')
    n = len(blocks)
    chosen = sorted(set(args.layers or [n // 3, 2 * n // 3, n - 1]))
    if any(i < 0 or i >= n for i in chosen):
        raise ValueError(f'Module layer indexes must be 0..{n - 1}.')
    cache, collected, handles = {}, {}, []

    def hook(key):
        def capture(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            x = x.detach().float()[0]
            x = x[-1] if args.pool == 'last' else x.mean(dim=0)
            cache[key] = x.cpu().numpy()
        return capture

    for i, (name, module) in sorted(blocks.items()):
        handles.append(module.register_forward_hook(hook(f'hidden_{i}')))
    suffixes = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    for name, module in model.named_modules():
        m = re.search(r'layers\.(\d+)\.', name)
        if m and int(m.group(1)) in chosen and name.split('.')[-1] in suffixes:
            key = f'module_{int(m.group(1))}_{name.split(".")[-1]}'
            handles.append(module.register_forward_hook(hook(key)))
    if len(handles) == n:
        raise ValueError('No supported projection modules found.')
    losses, counts, token_ids = [], [], []
    input_device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            for i, r in enumerate(rows):
                enc = tok(r['text'], return_tensors='pt', truncation=True,
                          max_length=args.max_length, add_special_tokens=True)
                ids = enc['input_ids'][0].tolist()
                if len(ids) < 2:
                    raise ValueError(f'Sample {r["id"]} has fewer than 2 tokens.')
                token_ids.append(ids)
                counts.append(len(ids) - 1)
                enc = {k: v.to(input_device) for k, v in enc.items()}
                cache.clear()
                out = model(**enc, labels=enc['input_ids'], use_cache=False,
                            output_hidden_states=False, return_dict=True)
                losses.append(float(out.loss))
                for key, val in cache.items():
                    collected.setdefault(key, []).append(val)
                del out
                print(f'{i + 1}/{len(rows)} loss={losses[-1]:.4f}', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if any(len(v) != len(rows) for v in collected.values()):
        raise ValueError('A hooked module did not execute once per sample.')
    hidden = np.stack([np.stack(collected.pop(f'hidden_{i}')) for i in range(n)], axis=1)
    meta = dict(model=args.model, tokenizer=args.tokenizer or args.model, pool=args.pool,
                max_length=args.max_length, rows=rows, token_ids=token_ids,
                module_layers=chosen, dtype=args.dtype, torch=torch.__version__,
                transformers=transformers.__version__, layer_index='zero-based block output',
                input_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, hidden=hidden, loss=np.array(losses), counts=np.array(counts),
                        metadata=np.array(json.dumps(meta)),
                        **{k: np.stack(v) for k, v in collected.items()})
    print('Saved', path)


def cosine(a, b):
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide((a * b).sum(axis=-1), den,
                     out=np.full(den.shape, np.nan), where=den > 1e-12)


def cka(x, y):
    x = x.astype(np.float64) - x.mean(axis=0)
    y = y.astype(np.float64) - y.mean(axis=0)
    k, l = x @ x.T, y @ y.T
    den = np.linalg.norm(k) * np.linalg.norm(l)
    return float((k * l).sum() / den) if den > 1e-12 else float('nan')


def heatmap(data, title, path, *, signed=False, limits=None, xticks=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 9))
    if limits is None:
        v = max(float(np.nanmax(np.abs(data))), 1e-8)
        limits = (-v, v) if signed else (0, v)
    im = ax.imshow(data, aspect='auto', interpolation='nearest',
                   cmap='RdBu_r' if signed else 'viridis', vmin=limits[0], vmax=limits[1])
    ax.set(title=title, xlabel='Layer (zero-based) / channel ID', ylabel='Sample index (CSV order)')
    if xticks is not None:
        step = max(1, len(xticks) // 16)
        ax.set_xticks(np.arange(0, len(xticks), step))
        ax.set_xticklabels(np.asarray(xticks)[::step], rotation=60)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def csv_file(path, fields, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def overview(rms_b, rms_a, relative, cos, layers, loss_delta, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    vmax = max(float(rms_b.max()), float(rms_a.max()), 1e-8)
    panels = [(rms_b, 'Before: hidden-state RMS', (0, vmax)),
              (rms_a, 'After: hidden-state RMS', (0, vmax)),
              (relative, 'Relative drift', None),
              (1 - cos, 'Directional drift (1 - cosine)', (0, 2))]
    for ax, (values, title, limits) in zip(axes.flat, panels):
        options = dict(vmin=limits[0], vmax=limits[1]) if limits else {}
        im = ax.imshow(values, aspect='auto', interpolation='nearest', cmap='viridis', **options)
        ax.set(title=title, xlabel='Layer (zero-based)', ylabel='Sample index')
        fig.colorbar(im, ax=ax)
    axes[1, 1].plot([r['layer'] for r in layers], [r['linear_cka'] for r in layers], marker='.')
    axes[1, 1].set(title='Linear CKA on fixed inputs', xlabel='Layer', ylabel='CKA', ylim=(0, 1.02))
    axes[1, 2].scatter(relative.mean(axis=1), loss_delta, s=16, alpha=.65)
    axes[1, 2].axhline(0, color='gray', linewidth=1)
    axes[1, 2].set(title='Drift vs. sentence NLL change', xlabel='Mean relative drift', ylabel='NLL after - before')
    fig.suptitle('Fixed-input activation comparison (descriptive, not causal)', fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare(args):
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(args.before, allow_pickle=False) as before, np.load(args.after, allow_pickle=False) as after:
        mb, ma = [json.loads(str(z['metadata'])) for z in (before, after)]
        for key in ('rows', 'token_ids', 'pool', 'max_length', 'module_layers'):
            if mb[key] != ma[key]:
                raise ValueError(f'Incompatible runs: {key} differs. Use identical inputs/tokenizer/options.')
        if set(before.files) != set(after.files) or before['hidden'].shape != after['hidden'].shape:
            raise ValueError('Architecture or extracted modules differ.')
        x, y = before['hidden'].astype(np.float64), after['hidden'].astype(np.float64)
        rms_b, rms_a = [np.sqrt((h * h).mean(axis=-1)) for h in (x, y)]
        scale = (0, max(float(rms_b.max()), float(rms_a.max())))
        heatmap(rms_b, 'Before: hidden-state RMS', dest / '01_before.png', limits=scale)
        heatmap(rms_a, 'After: hidden-state RMS', dest / '02_after.png', limits=scale)
        heatmap(rms_a - rms_b, 'After minus before: RMS change', dest / '03_rms_difference.png', signed=True)
        relative = np.linalg.norm(y - x, axis=-1) / np.maximum(np.linalg.norm(x, axis=-1), 1e-12)
        cos = cosine(x, y)
        heatmap(relative, 'Relative hidden-state drift', dest / '04_relative_drift.png')
        heatmap(1 - cos, 'Directional drift: 1 - cosine', dest / '05_directional_drift.png', limits=(0, 2))
        fields = ['sample_index', 'id', 'group', 'text', 'layer', 'loss_before', 'loss_after',
                  'loss_delta', 'relative_drift', 'cosine']
        output = []
        for i, r in enumerate(mb['rows']):
            for layer in range(x.shape[1]):
                output.append(dict(sample_index=i, id=r['id'], group=r.get('group', ''), text=r['text'],
                                   layer=layer, loss_before=before['loss'][i], loss_after=after['loss'][i],
                                   loss_delta=after['loss'][i] - before['loss'][i],
                                   relative_drift=relative[i, layer], cosine=cos[i, layer]))
        csv_file(dest / 'per_sample_layer.csv', fields, output)
        layers = [dict(layer=l, linear_cka=cka(x[:, l], y[:, l]),
                       mean_relative_drift=relative[:, l].mean()) for l in range(x.shape[1])]
        csv_file(dest / 'layer_summary.csv', list(layers[0]), layers)
        overview(rms_b, rms_a, relative, cos, layers, after['loss'] - before['loss'], dest / '00_overview.png')
        selected = {}
        for key in sorted(k for k in before.files if k.startswith('module_')):
            a, b = before[key], after[key]
            if a.shape != b.shape:
                raise ValueError(f'Shape mismatch in {key}')
            delta = b - a
            ids = np.argsort(np.mean(np.abs(delta), axis=0))[::-1][:args.top_k]
            selected[key] = ids.tolist()
            v = max(float(np.abs(a[:, ids]).max()), float(np.abs(b[:, ids]).max()), 1e-8)
            heatmap(a[:, ids], f'{key}: before', dest / f'{key}_before.png', signed=True, limits=(-v, v), xticks=ids)
            heatmap(b[:, ids], f'{key}: after', dest / f'{key}_after.png', signed=True, limits=(-v, v), xticks=ids)
            heatmap(delta[:, ids], f'{key}: signed pooled activation difference',
                    dest / f'{key}_difference.png', signed=True, xticks=ids)
        weighted = [float(np.average(z['loss'], weights=z['counts'])) for z in (before, after)]
        summary = dict(before=mb, after=ma, token_weighted_nll_before=weighted[0],
                       token_weighted_nll_after=weighted[1], nll_delta=weighted[1] - weighted[0],
                       selected_channels=selected,
                       warning='Activation drift does not establish forgetting or causality. '
                               'Top channels selected on this same sample set are exploratory.')
        (dest / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Saved heatmaps and CSV:', dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--model', required=True, help='Full HF checkpoint, not an unmerged adapter')
    e.add_argument('--tokenizer', help='Use the same tokenizer for both checkpoints')
    e.add_argument('--inputs', required=True)
    e.add_argument('--out', required=True, help='Output .npz filename')
    e.add_argument('--device', default='auto')
    e.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    e.add_argument('--pool', choices=['last', 'mean'], default='last')
    e.add_argument('--layers', type=int, nargs='+', help='Module layers; block states always include all layers')
    e.add_argument('--max-length', type=int, default=128)
    e.add_argument('--limit', type=int, default=100)
    c = sub.add_parser('compare')
    c.add_argument('--before', required=True)
    c.add_argument('--after', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--top-k', type=int, default=64)
    args = parser.parse_args()
    if getattr(args, 'top_k', 1) < 1 or getattr(args, 'max_length', 2) < 2 or getattr(args, 'limit', 0) < 0:
        parser.error('top-k >= 1, max-length >= 2, limit >= 0 required')
    (extract if args.command == 'extract' else compare)(args)


if __name__ == '__main__':
    main()
