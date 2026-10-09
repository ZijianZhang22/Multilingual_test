#!/usr/bin/env python3
"""Controlled English-dominant LM -> Chinese continual pretraining pilot."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARMS = ('zh_only', 'en_only', 'mixed')


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def document_split(text):
    # Exact duplicate text always goes to the same split, irrespective of article ID.
    digest = hashlib.sha256(text.strip().encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return digest, 'test' if bucket < 10 else 'val' if bucket < 20 else 'train'


def arm_languages(arm, count):
    if arm == 'mixed':
        if count % 2:
            raise ValueError('Mixed arm needs an even number of microbatches per optimizer step.')
        return ['en', 'zh'] * (count // 2)
    return [('zh' if arm == 'zh_only' else 'en')] * count


def bootstrap_delta(before, after, doc_ids, seed=42, repeats=1000):
    """Paired document-cluster bootstrap; mean remains token-weighted for equal blocks."""
    delta = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    unique = sorted(set(doc_ids))
    sums = np.array([delta[np.array(doc_ids) == doc].sum() for doc in unique])
    counts = np.array([doc_ids.count(doc) for doc in unique])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        selected = rng.integers(0, len(unique), size=len(unique))
        values.append(float(sums[selected].sum() / counts[selected].sum()))
    lo, hi = np.quantile(values, [.025, .975])
    return dict(delta=float(delta.mean()), ci95=[float(lo), float(hi)], documents=len(unique),
                blocks=len(delta), uncertainty='Paired bootstrap over held-out document clusters; one training seed.')


def prepare(args):
    from transformers import AutoTokenizer
    from datasets import load_dataset
    out = Path(args.out)
    tok = AutoTokenizer.from_pretrained(args.model)
    n_train = args.steps * args.micro_batch * args.grad_accum
    expected = dict(model=args.model, dataset='wikimedia/wikipedia', dataset_revision=args.data_revision,
                    configs=['20231101.en', '20231101.zh'], block_size=args.block_size,
                    train_blocks_per_language=n_train, eval_blocks=args.eval_blocks, data_seed=args.data_seed,
                    split='sha256(normalized exact article text) % 100: test <10, val <20, else train', version=1)
    manifest = out / 'data' / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['settings'] != expected:
            raise ValueError('Prepared data settings differ. Use a fresh --out.')
        for lang in ('en', 'zh'):
            with np.load(out / 'data' / f'{lang}.npz', allow_pickle=False) as z:
                if len(z['train']) != n_train or len(z['test']) != args.eval_blocks or len(z['val']) != args.eval_blocks:
                    raise ValueError('Incomplete data cache; use a fresh --out.')
        print('Reusing prepared data', flush=True)
        return
    provenance = {}
    for lang in ('en', 'zh'):
        arrays = {key: [] for key in ('train', 'val', 'test')}
        docs = {key: [] for key in arrays}
        targets = dict(train=n_train, val=args.eval_blocks, test=args.eval_blocks)
        seen = set()
        stream = load_dataset('wikimedia/wikipedia', f'20231101.{lang}', split='train',
                              streaming=True, revision=args.data_revision)
        stream = stream.shuffle(seed=args.data_seed, buffer_size=args.shuffle_buffer)
        for article in stream:
            text = article.get('text', '').strip()
            if not text:
                continue
            digest, split = document_split(text)
            if digest in seen:
                continue
            seen.add(digest)
            if len(arrays[split]) >= targets[split]:
                continue
            ids = tok(text, add_special_tokens=False, truncation=False)['input_ids']
            if tok.eos_token_id is not None:
                ids.append(tok.eos_token_id)
            # No block spans two documents; a document never crosses splits.
            cap = 64 if split == 'train' else 4
            for start in range(0, min(len(ids) - args.block_size + 1, cap * args.block_size), args.block_size):
                arrays[split].append(ids[start:start + args.block_size])
                docs[split].append(digest)
                if len(arrays[split]) >= targets[split]:
                    break
            if all(len(arrays[key]) == targets[key] for key in targets):
                break
        if any(len(arrays[key]) != targets[key] for key in targets):
            raise RuntimeError(f'Insufficient data for {lang}')
        for a in docs:
            for b in docs:
                if a != b and set(docs[a]) & set(docs[b]):
                    raise AssertionError('Document leakage')
        path = out / 'data' / f'{lang}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: np.asarray(v, dtype=np.int32) for k, v in arrays.items()})
        write_json(out / 'data' / f'{lang}_documents.json', docs)
        provenance[lang] = dict(blocks={k: len(v) for k, v in arrays.items()},
                                documents={k: len(set(v)) for k, v in docs.items()},
                                tokenized_blocks_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print('Prepared', lang, provenance[lang], flush=True)
    write_json(manifest, dict(settings=expected, provenance=provenance))


def evaluate(model, blocks, batch, device, amp):
    import torch
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(blocks), batch):
            x = torch.as_tensor(blocks[start:start + batch].astype(np.int64), device=device)
            with amp():
                logits = model(input_ids=x, use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction='none')
            values.extend(loss.reshape(len(x), -1).mean(dim=1).cpu().tolist())
            del logits, loss, x
    return values


def train(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    if device == 'cpu' and not args.cpu:
        raise RuntimeError('No CUDA GPU. Use --cpu only for a tiny smoke test; formal training needs a Pod.')
    bf16 = device == 'cuda' and torch.cuda.is_bf16_supported()
    amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if bf16 else contextlib.nullcontext
    print('Training', args.arm, 'on', device, 'bf16 autocast', bf16, flush=True)
    base = Path(args.out)
    out = base / args.arm
    if (out / 'metrics.json').exists() or any(out.glob('step_*')):
        raise ValueError('This arm already has results/checkpoints. Use a fresh --out; training resume is not supported.')
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                attn_implementation='sdpa').to(device)
    if args.block_size > getattr(model.config, 'max_position_embeddings', args.block_size):
        raise ValueError('block-size exceeds the model context limit')
    model.config.use_cache = False
    if not args.cpu:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    tok = AutoTokenizer.from_pretrained(args.model)
    datasets = {}
    for lang in ('en', 'zh'):
        with np.load(base / 'data' / f'{lang}.npz', allow_pickle=False) as z:
            datasets[lang] = {key: z[key].copy() for key in ('train', 'val', 'test')}
    # Independent identical streams across arms: mixed arm consumes prefixes of each stream.
    for j, lang in enumerate(('en', 'zh')):
        rng = np.random.default_rng(args.data_seed + j)
        datasets[lang]['train'] = datasets[lang]['train'][rng.permutation(len(datasets[lang]['train']))]
    cursors = dict(en=0, zh=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    marks = sorted(set([0, args.steps, *args.eval_marks, *args.save_marks]))
    saves = set([args.steps, *args.save_marks])
    metrics = []
    train_log = []
    def measure(step):
        entry = dict(step=step, tokens_seen=step * args.micro_batch * args.grad_accum * args.block_size,
                     prediction_tokens_seen=step * args.micro_batch * args.grad_accum * (args.block_size - 1),
                     language_tokens_seen={k: cursors[k] * args.block_size for k in cursors}, losses={})
        for lang in ('en', 'zh'):
            entry['losses'][lang] = {split: evaluate(model, datasets[lang][split], args.eval_batch, device, amp)
                                      for split in ('val', 'test')}
        if any(not np.isfinite(v).all() for splits in entry['losses'].values() for v in splits.values()):
            raise FloatingPointError('Nonfinite evaluation loss')
        metrics.append(entry)
        write_json(out / 'metrics.json', metrics)
        print('EVAL', step, {lang: {split: round(float(np.mean(v)), 5) for split, v in splits.items()}
                             for lang, splits in entry['losses'].items()}, flush=True)
        if step in saves and step:
            checkpoint = out / f'step_{step:06d}'
            model.save_pretrained(checkpoint, safe_serialization=True)
            tok.save_pretrained(checkpoint)
        model.train()
    measure(0)
    languages = arm_languages(args.arm, args.grad_accum)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        warmup = max(1, args.warmup_steps)
        lr = args.lr * min(1., step / warmup) if args.warmup_steps else args.lr
        for group in optimizer.param_groups:
            group['lr'] = lr
        losses = []
        for lang in languages:
            start = cursors[lang]
            arr = datasets[lang]['train'][start:start + args.micro_batch]
            if len(arr) != args.micro_batch:
                raise RuntimeError('Training token budget exceeded; data must not cycle silently.')
            cursors[lang] += args.micro_batch
            x = torch.as_tensor(arr.astype(np.int64), device=device)
            with amp():
                result = model(input_ids=x, labels=x, use_cache=False)
                loss = result.loss / args.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            losses.append(float(result.loss.detach()))
            del result, loss, x
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        record = dict(step=step, loss=float(np.mean(losses)), lr=lr, gradient_norm=float(norm))
        train_log.append(record)
        print('TRAIN', args.arm, record, flush=True)
        if step in marks:
            measure(step)
    write_json(out / 'training_log.json', train_log)
    write_json(out / 'settings.json', dict(arguments=vars(args), device=device, bf16_autocast=bf16, model_commit=getattr(model.config, '_commit_hash', None),
        torch=torch.__version__, parameters=sum(p.numel() for p in model.parameters()),
        completed=True, note='Independent restart from the same base model in each arm.'))


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(args.out)
    all_rows, final = [], {}
    reference = None
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for arm in args.arms:
        metrics = json.loads((root / arm / 'metrics.json').read_text())
        baseline = metrics[0]['losses']
        if reference is None:
            reference = baseline
        elif any(not np.allclose(baseline[lang]['test'], reference[lang]['test'], atol=1e-5, rtol=1e-5) for lang in ('en', 'zh')):
            raise ValueError('Arms have different step-0 losses; check their model/data/settings before comparing')
        for lang in ('en', 'zh'):
            doc_ids = json.loads((root / 'data' / f'{lang}_documents.json').read_text())['test']
            points = []
            for entry in metrics:
                stats = bootstrap_delta(baseline[lang]['test'], entry['losses'][lang]['test'], doc_ids)
                row = dict(arm=arm, step=entry['step'], tokens_seen=entry['tokens_seen'], language=lang,
                           nll=float(np.mean(entry['losses'][lang]['test'])), delta=stats['delta'],
                           ci_low=stats['ci95'][0], ci_high=stats['ci95'][1], documents=stats['documents'])
                all_rows.append(row)
                points.append(row)
            ax = axes[0 if lang == 'en' else 1]
            ax.plot([p['tokens_seen'] for p in points], [p['delta'] for p in points], marker='o', label=arm)
            ax.fill_between([p['tokens_seen'] for p in points], [p['ci_low'] for p in points],
                            [p['ci_high'] for p in points], alpha=.12)
            final.setdefault(arm, {})[lang] = dict(stats, baseline_nll=float(np.mean(baseline[lang]['test'])),
                final_nll=float(np.mean(metrics[-1]['losses'][lang]['test'])),
                perplexity_ratio=math.exp(min(stats['delta'], 100)))
        en, zh = final[arm]['en'], final[arm]['zh']
        if en['ci95'][0] > 0 and zh['ci95'][1] < 0:
            verdict = 'Old-language NLL regression with new-language improvement'
        elif en['ci95'][0] > 0 and zh['ci95'][0] > 0:
            verdict = 'Both languages worsen: check training instability/domain shift'
        else:
            verdict = 'No clear paired evidence of old-language regression plus new-language gain'
        final[arm]['verdict'] = verdict
        final[arm]['large_regression_flag'] = en['delta'] >= args.large_delta and en['ci95'][0] > 0 and zh['ci95'][1] < 0
        axes[2].scatter(-zh['delta'], en['delta'], label=arm)
        axes[2].annotate(arm, (-zh['delta'], en['delta']))
    control = {}
    if 'zh_only' in final:
        for other in ('en_only', 'mixed'):
            if other in final:
                a = json.loads((root / 'zh_only' / 'metrics.json').read_text())
                b = json.loads((root / other / 'metrics.json').read_text())
                if a[-1]['tokens_seen'] != b[-1]['tokens_seen']:
                    raise ValueError('Controls have unmatched token budgets')
                docs = json.loads((root / 'data' / 'en_documents.json').read_text())['test']
                control[other] = bootstrap_delta(b[-1]['losses']['en']['test'], a[-1]['losses']['en']['test'], docs)
    for i, title in enumerate(('English held-out NLL change', 'Chinese held-out NLL change')):
        axes[i].set(title=title, xlabel='Training input tokens', ylabel='NLL - baseline')
        axes[i].axhline(0, color='gray', lw=1)
        axes[i].legend()
    axes[2].set(title='Retention / new-language gain', xlabel='Chinese NLL improvement', ylabel='English NLL regression')
    axes[2].axhline(0, color='gray', lw=1)
    axes[2].axvline(0, color='gray', lw=1)
    fig.tight_layout()
    fig.savefig(root / 'forgetting_curves.png', dpi=180)
    plt.close(fig)
    with (root / 'evaluation_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    write_json(root / 'conclusions.json', dict(arms=final, english_zh_only_minus_control=control,
        large_delta_threshold=args.large_delta,
        warning='Threshold is an exploratory flag, not a universal definition of catastrophic forgetting. '
                'One seed and Wikipedia NLL alone do not establish broad ability loss. '
                'CIs quantify evaluation-document variation, not training-seed variation. '
                'Mixed arm matches total tokens, not Chinese token exposure.'))
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


def run_logged(command, path):
    from importlib import util
    spec = util.spec_from_file_location('heatmap_runner', ROOT.parent / 'activation_heatmaps' / 'run_suite.py')
    mod = util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.run_logged(command, path)


def run(args):
    import torch
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for the formal run. --cpu is only for tiny smoke tests.')
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'invocation.json').exists():
        raise ValueError('Output already contains an invocation. Use a fresh --out to prevent mixed experiments.')
    write_json(root / 'invocation.json', vars(args))
    common = ['--model', args.model, '--out', str(root), '--steps', str(args.steps),
              '--block-size', str(args.block_size), '--micro-batch', str(args.micro_batch),
              '--grad-accum', str(args.grad_accum), '--eval-blocks', str(args.eval_blocks),
              '--eval-batch', str(args.eval_batch), '--lr', str(args.lr), '--seed', str(args.seed),
              '--data-seed', str(args.data_seed), '--data-revision', args.data_revision,
              '--shuffle-buffer', str(args.shuffle_buffer), '--cpu-threads', str(args.cpu_threads),
              '--weight-decay', str(args.weight_decay), '--clip-norm', str(args.clip_norm),
              '--warmup-steps', str(args.warmup_steps)]
    common += ['--eval-marks', *map(str, args.eval_marks)]
    common += ['--save-marks', *map(str, args.save_marks)]
    if args.cpu:
        common += ['--cpu']
    run_logged([sys.executable, str(Path(__file__).resolve()), 'prepare', *common], root / 'prepare.log')
    for arm in args.arms:
        run_logged([sys.executable, str(Path(__file__).resolve()), 'train', *common, '--arm', arm], root / f'{arm}.log')
    report(args)
    if not args.skip_heatmaps:
        script = ROOT.parent / 'activation_heatmaps' / 'run_suite.py'
        for arm in args.arms:
            for step in sorted(set([args.steps, *args.save_marks])):
                heatout = root / 'heatmaps' / arm / f'step_{step:06d}'
                command = [sys.executable, str(script), '--before', args.model,
                           '--after', str(root / arm / f'step_{step:06d}'), '--tokenizer', args.model,
                           '--pool', 'all', '--out', str(heatout), '--limit', str(args.heatmap_limit)]
                if args.cpu:
                    command += ['--device', 'cpu', '--dtype', 'float32']
                run_logged(command, root / f'heatmaps_{arm}_{step}.log')
    # Archive only reviewable results, never weights or large raw activation/data arrays.
    archive = root / 'route_b_results.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.csv', '.png', '.html', '.log', '.md'}:
                if any(part.startswith('step_') for part in path.relative_to(root).parts) and 'heatmaps' not in path.relative_to(root).parts:
                    continue
                z.write(path, path.relative_to(root))
    print('DONE. Results:', archive, '\nCheckpoints/raw activations remain on the Pod and are NOT in this ZIP.', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['run', 'prepare', 'train', 'report'])
    p.add_argument('--model', default='HuggingFaceTB/SmolLM2-360M')
    p.add_argument('--out', default='route_b_results/smol360_seed0')
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--arm', choices=ARMS, default='zh_only')
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--block-size', type=int, default=256)
    p.add_argument('--micro-batch', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--eval-blocks', type=int, default=128)
    p.add_argument('--eval-batch', type=int, default=2)
    p.add_argument('--eval-marks', nargs='*', type=int, default=[32, 64, 128])
    p.add_argument('--save-marks', nargs='*', type=int, default=[64])
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--clip-norm', type=float, default=1.)
    p.add_argument('--warmup-steps', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data-seed', type=int, default=1234)
    p.add_argument('--data-revision', default='b04c8d1ceb2f5cd4588862100d08de323dccfbaa')
    p.add_argument('--shuffle-buffer', type=int, default=256)
    p.add_argument('--large-delta', type=float, default=.5)
    p.add_argument('--heatmap-limit', type=int, default=100)
    p.add_argument('--skip-heatmaps', action='store_true')
    p.add_argument('--cpu', action='store_true', help='Only for tiny real-model smoke tests')
    p.add_argument('--cpu-threads', type=int, default=4)
    return p


def main():
    p = parser(); args = p.parse_args()
    if min(args.steps, args.micro_batch, args.grad_accum, args.eval_blocks, args.eval_batch, args.cpu_threads) < 1 or args.block_size < 2:
        p.error('Positive budgets and block-size >=2 required')
    if any(mark < 1 or mark > args.steps for mark in [*args.eval_marks, *args.save_marks]):
        p.error('Evaluation/save marks must lie in 1..steps; override them for smoke tests')
    if 'mixed' in args.arms or args.arm == 'mixed':
        if args.grad_accum % 2:
            p.error('grad-accum must be even for exact 50/50 mixed microbatches')
    if args.lr <= 0 or args.warmup_steps < 0 or args.heatmap_limit < 2 or args.shuffle_buffer < 1 or args.large_delta < 0:
        p.error('lr >0, warmup-steps >=0, heatmap-limit >=2 required')
    {'run': run, 'prepare': prepare, 'train': train, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
