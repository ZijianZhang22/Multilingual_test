#!/usr/bin/env python3
"""Sequential model extraction, PNG generation, HTML gallery and result archive."""
import argparse
import hashlib
import html
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PAIRS = {
    'tiny_zh': ('TinyLlama/TinyLlama_v1.1', 'TinyLlama/TinyLlama_v1.1_chinese'),
    'smol135_ru': ('HuggingFaceTB/SmolLM2-135M', 'nyuuzyou/SmolLM2-135M-Eagle'),
    'smol360_ru': ('HuggingFaceTB/SmolLM2-360M', 'nyuuzyou/SmolLM2-360M-Eagle'),
}


def valid_extraction(path, item, label):
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            expected = dict(pooling_version=2, model=item[label], tokenizer=item['tokenizer'],
                            input_sha256=item['input_sha256'], pool=item['pool'],
                            dtype=item['dtype'], max_length=item['max_length'])
            return (all(meta.get(k) == v for k, v in expected.items())
                    and (item['module_layers'] is None or sorted(set(item['module_layers'])) == meta.get('module_layers'))
                    and len(z['loss']) == len(meta['rows'])
                    and z['hidden'].shape[0] == len(meta['rows'])
                    and np.isfinite(z['loss']).all())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def run_logged(command, log):
    print('RUN:', ' '.join(command), flush=True)
    with log.open('w', encoding='utf-8') as stream:
        stream.write('Command: ' + json.dumps(command) + '\n')
        stream.flush()
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            status = proc.wait()
    if status:
        raise RuntimeError(f'Command failed (exit {status}); see {log}')


def build_plan(args):
    root = Path(__file__).resolve().parent
    from compare_activations import records
    inputs = Path(args.inputs or root / 'inputs_100_en_diverse.jsonl').resolve()
    rows = records(inputs, args.limit)
    row_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if args.before or args.after:
        if not args.before or not args.after:
            raise ValueError('--before and --after must both be supplied.')
        pairs = [('custom', args.before, args.after)]
    else:
        names = list(PAIRS) if args.pair == 'all' else [args.pair]
        pairs = [(name, *PAIRS[name]) for name in names]
    out = Path(args.out or root.parents[1] / 'activation_heatmap_results_v2').resolve()
    plan = []
    for name, before, after in pairs:
        tokenizer = args.tokenizer or before
        item = dict(pair=name, before=before, after=after, tokenizer=tokenizer,
                    input_sha256=row_hash, pool=args.pool, dtype=args.dtype,
                    max_length=args.max_length, module_layers=args.layers)
        directory = out / name
        steps = []
        for label, model in [('before', before), ('after', after)]:
            cmd = [sys.executable, str(root / 'compare_activations.py'), 'extract',
                   '--model', model, '--tokenizer', tokenizer, '--inputs', str(inputs),
                   '--out', str(directory / f'{label}.npz'), '--pool', args.pool,
                   '--dtype', args.dtype, '--device', args.device, '--limit', str(args.limit),
                   '--max-length', str(args.max_length)]
            if args.layers:
                cmd += ['--layers', *map(str, args.layers)]
            steps.append((label, cmd))
        pools = ('mean', 'last_nonpunct', 'last') if args.pool == 'all' else (args.pool,)
        for pool in pools:
            target = directory / pool if args.pool == 'all' else directory
            steps.append((f'compare_{pool}', [sys.executable, str(root / 'compare_activations.py'), 'compare',
                             '--before', str(target / 'before.npz'), '--after', str(target / 'after.npz'),
                             '--out', str(target / 'comparison'), '--top-k', str(args.top_k)]))
        plan.append((item, directory, steps))
    return out, inputs, plan


def gallery(out, plan):
    sections = []
    for item, directory, _ in plan:
        comp = directory / 'comparison'
        summary = json.loads((comp / 'summary.json').read_text())
        images = sorted(comp.glob('*.png'))
        cards = []
        for image in images:
            rel = html.escape(image.relative_to(out).as_posix(), quote=True)
            cards.append(f'<figure><a href="{rel}"><img loading="lazy" src="{rel}"></a>'
                         f'<figcaption>{html.escape(image.name)}</figcaption></figure>')
        sections.append(f'<section><h2>{html.escape(item["pair"])} / {html.escape(item.get("pool", "last"))}</h2>'
                        f'<p>{html.escape(item["before"])} → {html.escape(item["after"])}</p>'
                        f'<p>Token-weighted NLL: {summary["token_weighted_nll_before"]:.4f} → '
                        f'{summary["token_weighted_nll_after"]:.4f}</p>'
                        f'<div class="grid">{"".join(cards)}</div></section>')
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Activation comparison</title><style>body{font-family:sans-serif;margin:24px}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:12px}'
        'figure{margin:0}img{width:100%}figcaption{overflow-wrap:anywhere}section{margin-bottom:32px}</style>'
        '<h1>Fixed-input activation comparisons</h1>'
        '<p>Exploratory activation differences; these figures alone do not establish forgetting or causality.</p>'
        + ''.join(sections), encoding='utf-8')
    archive = out / 'activation_heatmaps_results.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            # Include only this invocation's selected pairs; omit large raw arrays.
            if not path.is_file() or path == archive or path.suffix == '.npz':
                continue
            if path.parent == out or any(path.is_relative_to(d.parent if item.get('pool') in ('mean', 'last_nonpunct', 'last') and d.name == item.get('pool') else d) for item, d, _ in plan):
                z.write(path, path.relative_to(out))
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair', choices=['all', *PAIRS], default='tiny_zh')
    p.add_argument('--before', help='Own full checkpoint before adaptation')
    p.add_argument('--after', help='Own full checkpoint after adaptation')
    p.add_argument('--tokenizer')
    p.add_argument('--inputs')
    p.add_argument('--out')
    p.add_argument('--pool', choices=['last', 'mean', 'last_nonpunct', 'all'], default='all')
    p.add_argument('--dtype', choices=['auto', 'bfloat16', 'float16', 'float32'], default='auto')
    p.add_argument('--device', default='auto')
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--top-k', type=int, default=64)
    p.add_argument('--layers', nargs='+', type=int)
    p.add_argument('--resume', action='store_true', help='Reuse complete extraction with matching manifest')
    p.add_argument('--dry-run', action='store_true', help='Print commands without importing torch or downloading')
    args = p.parse_args()
    if args.limit < 0 or args.max_length < 2 or args.top_k < 1:
        p.error('limit >=0, max-length >=2, top-k >=1 required')
    if args.dtype == 'auto':
        if args.dry_run:
            args.dtype = 'bfloat16'  # GPU preview only; actual execution detects CPU/capability.
        else:
            import torch
            args.dtype = ('bfloat16' if torch.cuda.is_bf16_supported() else 'float16') if torch.cuda.is_available() and args.device != 'cpu' else 'float32'
    try:
        out, inputs, plan = build_plan(args)
    except ValueError as exc:
        p.error(str(exc))
    if args.dry_run:
        for item, directory, steps in plan:
            for _, command in steps:
                print(' '.join(command))
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / 'invocation.json').write_text(json.dumps(dict(arguments=vars(args), inputs=str(inputs),
        started_utc=datetime.now(timezone.utc).isoformat()), indent=2))
    for item, directory, steps in plan:
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / 'run_manifest.json'
        previous = json.loads(manifest.read_text()) if manifest.exists() else None
        if args.resume and previous is not None and previous != item:
            raise ValueError(f'Resume settings changed for {item["pair"]}; use a different --out or omit --resume.')
        reuse = args.resume and previous == item
        manifest.write_text(json.dumps(item, indent=2))
        for label, cmd in steps:
            pools = ('mean', 'last_nonpunct', 'last') if item['pool'] == 'all' else (item['pool'],)
            complete = (all(valid_extraction(directory / pool / f'{label}.npz', dict(item, pool=pool), label) for pool in pools)
                        if item['pool'] == 'all' else valid_extraction(directory / f'{label}.npz', item, label)) if not label.startswith('compare') else False
            if reuse and complete:
                print('REUSE:', directory / f'{label}.npz', flush=True)
                continue
            run_logged(cmd, directory / f'{label}.log')
    gallery_plan = []
    for item, directory, steps in plan:
        if item['pool'] == 'all':
            gallery_plan.extend((dict(item, pool=pool), directory / pool, steps) for pool in ('mean', 'last_nonpunct', 'last'))
        else:
            gallery_plan.append((item, directory, steps))
    archive = gallery(out, gallery_plan)
    print(f'\nDONE\nImages and CSV: {out}\nGallery: {out / "index.html"}\nDownload archive: {archive}')


if __name__ == '__main__':
    main()
