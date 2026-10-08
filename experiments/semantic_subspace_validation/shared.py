"""Pure utilities for 0.5B semantic-subspace validation. Python >=3.10."""
from __future__ import annotations
import csv, json, random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    columns = list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def orth(q):
    q, _ = torch.linalg.qr(torch.as_tensor(q).float(), mode='reduced')
    return q

def random_subspace(dim, rank, seed):
    g = torch.Generator().manual_seed(int(seed))
    return orth(torch.randn(dim, rank, generator=g))

def drift_partition(qd, qi, ks=(16, 32)):
    qd, qi = orth(qd), orth(qi)
    mat = qd.T @ qi @ qi.T @ qd
    mat = (mat + mat.T) / 2
    evals, v = torch.linalg.eigh(mat)
    inds = torch.argsort(evals, descending=True)
    ordered = qd @ v[:, inds]
    ordered = orth(ordered)
    cos2 = ((ordered.T @ qi) ** 2).sum(dim=1)
    ids = torch.argsort(cos2, descending=True)
    ordered, cos2 = ordered[:, ids], cos2[ids]
    out = {}
    for k in ks:
        if k * 2 > qd.shape[1]:
            raise ValueError(f'Cannot extract disjoint top/bottom {k} from rank={qd.shape[1]}')
        out[f'top{k}'] = ordered[:, :k]
        out[f'bottom{k}'] = ordered[:, -k:]
    return out, cos2

def make_bases(subspace_file, train_x=None, rank=64, seed=2026):
    payload = torch.load(subspace_file, map_location='cpu', weights_only=False)
    q = {k: orth(v) for k, v in payload['subspaces'].items()
         if not k.startswith('random_')}
    if 'isr_multiclass' not in q or 'drift' not in q:
        raise ValueError('core_subspaces.pt needs drift and isr_multiclass.')
    q.update(drift_partition(q['drift'], q['isr_multiclass'])[0])
    dim = q['drift'].shape[0]
    if train_x is not None:
        xc = train_x.float() - train_x.float().mean(0)
        rank_pca = min(rank, min(xc.shape))
        _, _, v = torch.pca_lowrank(xc, q=rank_pca, center=False, niter=6)
        q['pca64'] = orth(v[:, :rank_pca])
    for k in (16, 32, rank):
        q[f'random{k}'] = random_subspace(dim, k, seed + k)
    for j in range(4):
        q[f'random32_draw{j}'] = random_subspace(dim, 32, seed + 400 + j)
    if len({v.shape[0] for v in q.values()}) != 1:
        raise ValueError('Incompatible hidden sizes across bases')
    return q, payload

def normalized(x):
    x = torch.as_tensor(x).float()
    return torch.nn.functional.normalize(x, dim=-1, eps=1e-8)

def conditional_pairs(rows, lang_a, lang_b, maximum=256, seed=91):
    grouped = defaultdict(dict)
    for idx, row in enumerate(rows):
        grouped[str(row['pair_id'])][row['language']] = idx
    pairs = [(group[lang_a], group[lang_b]) for group in grouped.values()
             if lang_a in group and lang_b in group]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:min(maximum, len(pairs))]
