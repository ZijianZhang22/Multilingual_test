#!/usr/bin/env python3
"""Run semantic validation on EXISTING Qwen2.5-7B EN->ZH checkpoint files.

Read-only wrt weights/subspaces. Never trains, never downloads Wikipedia, never
runs Step6/Step7 again. Uses XNLI test for held-out probe/retrieval results,
then forward-only behavioral interventions on the adapted model.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from shared import write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def check_ckpt(path):
    path = Path(path)
    if not (path/'config.json').is_file():
        return False
    idx = path/'model.safetensors.index.json'
    if idx.is_file():
        try:
            shards = set(json.loads(idx.read_text())['weight_map'].values())
            return bool(shards) and all((path/s).is_file() and (path/s).stat().st_size > 0
                                        for s in shards)
        except (OSError, KeyError, ValueError, TypeError):
            return False
    return (path/'model.safetensors').is_file() and (path/'model.safetensors').stat().st_size > 0


def call(items, dry_run=False):
    cmd = [str(x) for x in items]
    print('\n>>> ' + ' '.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--anchor_checkpoint', default='replication_runs/qwen25_7b_a100_fresh_fullzh/seed0/anchor')
    ap.add_argument('--adapted_checkpoint', default='replication_runs/qwen25_7b_a100_fresh_fullzh/seed0/lr_4e-05/adapted')
    ap.add_argument('--analysis_dir', default='replication_runs/qwen25_7b_a100_fresh_fullzh_analysis/seed0/lr_4e-05')
    ap.add_argument('--out_dir', default='replication_runs/qwen25_7b_a100_semantic_validation/seed0/lr_4e-05')
    ap.add_argument('--n_examples', type=int, default=18,
                    help='Held-out causal examples; preliminary 18, more robust 72+. A larger run should use a NEW --out_dir.')
    ap.add_argument('--max_pairs', type=int, default=256)
    ap.add_argument('--extract_batch', type=int, default=1)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--through', choices=['probes', 'causal'], default='causal')
    ap.add_argument('--dry_run', action='store_true')
    a = ap.parse_args()
    if a.n_examples < 9 or a.max_pairs < 8 or a.extract_batch < 1:
        ap.error('Need n_examples>=9, max_pairs>=8, extract_batch>=1.')
    anchor = (ROOT/a.anchor_checkpoint).resolve()
    adapted = (ROOT/a.adapted_checkpoint).resolve()
    analysis = (ROOT/a.analysis_dir).resolve()
    output = (ROOT/a.out_dir).resolve()
    sub = analysis/'subspaces'
    core = sub/'core_subspaces.pt'
    probe = sub/'data/xnli_probe.jsonl'
    train_features = sub/'features/anchor_probe.pt'
    aligned = output/'data/xnli_aligned_test.jsonl'
    aligned_features = output/'features/anchor_aligned_test.pt'
    probedir = output/'probes_and_retrieval'
    causal_dir = output/'causal_semantics'
    bases = probedir/'validated_bases.pt'

    if not a.dry_run:
        required = [core, probe, train_features]
        if not check_ckpt(anchor) or not check_ckpt(adapted):
            raise FileNotFoundError(
                f'7B checkpoints unavailable/incomplete: anchor={anchor}; adapted={adapted}. '
                'This runner NEVER retrains; verify files on the original Pod.')
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError('7B subspace fit files are missing: ' + '; '.join(missing))
        import torch
        metadata = torch.load(core, map_location='cpu', weights_only=False)
        if not 1 <= int(metadata.get('layer', -1)) <= 28:
            raise ValueError(f'Invalid 7B subspace layer: {metadata.get("layer")}')
        config = json.loads((anchor/'config.json').read_text())
        if int(config.get('hidden_size', -1)) != int(metadata['hidden_dim']):
            raise ValueError('Anchor hidden size does not match subspace dimensions.')
        if metadata['hidden_dim'] < 1000:
            raise ValueError('Unexpected hidden dimension for 7B; is this a 0.5B subspace file?')
        if len(set((metadata['subspaces']['drift'].shape[0],
                    metadata['subspaces']['isr_multiclass'].shape[0]))) != 1:
            raise ValueError('Inconsistent subspace shapes.')
        output.mkdir(parents=True, exist_ok=True)
        config_path = output/'run_config.json'
        settings = dict(anchor_checkpoint=str(anchor), adapted_checkpoint=str(adapted),
                        analysis_dir=str(analysis), n_examples=a.n_examples,
                        max_pairs=a.max_pairs, seed=a.seed, extract_batch=a.extract_batch,
                        semantic_protocol='Qwen2.5-7B: held-out XNLI probes, retrieval, forward-only causal edits',
                        note='Drift basis may use some XNLI probe_test inputs during its unsupervised fitting; not fully inductive. '
                             'Semantic task labels are held out, but avoid overclaiming generalization.')
        if config_path.is_file():
            prior = json.loads(config_path.read_text())
            if prior != settings:
                raise RuntimeError(f'Changed config in existing output folder {output}; use a different --out_dir.')
        else:
            write_json(config_path, settings)
        print(f'[7B] anchor={anchor}\n[7B] adapted={adapted}\n[7B] layer={metadata["layer"]}', flush=True)

    layer = 23
    if not a.dry_run:
        layer = int(metadata['layer'])
    if not aligned.is_file():
        call([sys.executable, ROOT/'invariance/prepare_aligned_xnli.py',
              '--languages', 'en','zh','fr','de','es', '--split','test',
              '--n_examples',max(a.max_pairs,256), '--seed',a.seed,
              '--out_file',aligned], a.dry_run)
    if not aligned_features.is_file():
        call([sys.executable, ROOT/'invariance/extract_hidden.py',
              '--checkpoint',anchor, '--data_file',aligned,
              '--out_file',aligned_features,
              '--layers',layer, '--batch_size',a.extract_batch],a.dry_run)
    if not ((probedir/'probe_results.csv').is_file()
            and (probedir/'retrieval_results.csv').is_file() and bases.is_file()):
        call([sys.executable,HERE/'probes_retrieval.py',
              '--subspace_file',core,'--probe_features',train_features,
              '--aligned_features',aligned_features,'--aligned_data',aligned,
              '--out_dir',probedir,'--max_pairs',a.max_pairs,'--seed',a.seed],a.dry_run)
    else:
        print('[resume] probes and retrieval already complete', flush=True)

    if a.through == 'causal':
        if not (causal_dir/'causal_summary.csv').is_file():
            call([sys.executable,HERE/'causal_semantics.py',
                  '--checkpoint',adapted,'--subspace_file',core,
                  '--bases_file',bases,'--probe_data',probe,
                  '--n_examples',a.n_examples,'--seed',a.seed,
                  '--out_dir',causal_dir],a.dry_run)
        else:
            print('[resume] causal semantics already complete',flush=True)
    if not a.dry_run and a.through == 'causal':
        call([sys.executable,HERE/'summarize.py','--run_dir',analysis,'--out_dir',output])
    print(f'[done] requested 7B validation through {a.through}: {output}', flush=True)


if __name__ == '__main__':
    main()
