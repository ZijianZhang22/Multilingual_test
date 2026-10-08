"""Shared numerical utilities for literature-inspired multilingual measurements.

Pure torch helpers live here so they can be tested without loading an LLM.
Transformer checkpoints are loaded lazily by load_model().
"""
import csv
import json
import math
from pathlib import Path

import torch


def rows_from_jsonl(path, languages=None):
    rows = []
    allowed = set(languages) if languages else None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if allowed is not None and row.get("language") not in allowed:
                continue
            if "text" in row:
                text = row["text"]
            elif "premise" in row and "hypothesis" in row:
                text = row["premise"] + "\n\n" + row["hypothesis"]
            else:
                raise ValueError("Each JSONL row needs text or premise+hypothesis")
            if not isinstance(text, str) or not text.strip():
                continue
            rows.append({**row, "text": text})
    if not rows:
        raise ValueError(f"No usable input rows: {path}")
    return rows


def ensure_disjoint(fit_rows, eval_rows):
    """Reject known overlapping IDs and identical text, even across filenames."""
    def ids(rows):
        return {str(r["example_id"]) for r in rows if r.get("example_id") is not None}
    overlap = ids(fit_rows) & ids(eval_rows)
    if overlap:
        raise ValueError(f"Fit/eval IDs overlap ({len(overlap)}); e.g. {next(iter(overlap))}")
    texts = {r["text"] for r in fit_rows}
    overlap_texts = texts & {r["text"] for r in eval_rows}
    if overlap_texts:
        raise ValueError(f"Fit/eval input texts overlap ({len(overlap_texts)})")


def save_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_model(checkpoint, device, *, bf16=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dtype = torch.bfloat16 if bf16 and str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    model.config.use_cache = False
    return model, tok


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise TypeError("Expected a Qwen/Llama-style model.model.layers decoder")


def hidden_from_output(output):
    return output[0] if isinstance(output, tuple) else output


def with_hidden(output, replacement):
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    return replacement


def input_batch(tokenizer, texts, max_length, device, *, labels=False):
    e = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt")
    e = {k: v.to(device) for k, v in e.items()}
    if labels:
        lab = e["input_ids"].clone()
        lab[e["attention_mask"] == 0] = -100
        e["labels"] = lab
    return e


def orthonormal(q):
    if q.numel() == 0:
        raise ValueError("Empty basis")
    result, _ = torch.linalg.qr(q.float(), mode="reduced")
    return result


def affine_project(hidden, mean_in, mean_basis, q, *, mode):
    """Three Chang et al. variants; hidden may be [batch, seq, dim]."""
    if mode == "same":
        mu = mean_in
    elif mode == "cross":
        mu = mean_basis
    elif mode == "cross_recenter":
        mu = mean_in
    else:
        raise ValueError(mode)
    q = q.to(device=hidden.device, dtype=torch.float32)
    mu = mu.to(device=hidden.device, dtype=torch.float32)
    h = hidden.float()
    return (mu + ((h - mu) @ q) @ q.T).to(dtype=hidden.dtype)


def mean_pairwise_cosine(a, b):
    """Exact mean of all cross-set cosine similarities, no NxM matrix."""
    a = torch.nn.functional.normalize(a.float(), dim=-1, eps=1e-8)
    b = torch.nn.functional.normalize(b.float(), dim=-1, eps=1e-8)
    return float(a.mean(0).dot(b.mean(0)))


def linear_cka(x, y):
    """Requires ROW-aligned examples; do not use on unrelated token samples."""
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("CKA needs >=2 matched examples")
    x = x.float() - x.float().mean(0)
    y = y.float() - y.float().mean(0)
    numerator = (x.T @ y).pow(2).sum()
    denominator = torch.linalg.norm(x.T @ x) * torch.linalg.norm(y.T @ y)
    return float((numerator / denominator.clamp_min(1e-15)).item())


def principal_overlap(a, b):
    a, b = orthonormal(a), orthonormal(b)
    singular = torch.linalg.svdvals(a.T @ b).clamp(0, 1)
    return float(singular.square().mean())


def fit_affine(x, target_variance=0.90, max_rank=256, seed=0):
    """Approximate truncated PCA; report if requested variance is NOT reached.

    Explained variance uses FULL centered input Frobenius energy as denominator.
    The reported rank is never presented as 90% unless threshold was attained.
    """
    x = x.float().cpu()
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError("Need >=3 token vectors with shape [n, d]")
    mu = x.mean(0)
    xc = x - mu
    total = xc.square().sum()
    if total <= 0:
        raise ValueError("Feature variance is zero")
    q = min(max_rank + 8, x.shape[0] - 1, x.shape[1])
    torch.manual_seed(seed)
    _, singular, v = torch.pca_lowrank(xc, q=q, center=False, niter=4)
    cum = singular.square().cumsum(0) / total
    reached = bool(cum[-1] >= target_variance)
    n = int((cum >= target_variance).nonzero()[0].item() + 1) if reached else min(max_rank, q)
    n = min(n, max_rank)
    reached = bool(cum[n - 1] >= target_variance)
    return mu, orthonormal(v[:, :n]), {
        "rank": n, "explained_variance": float(cum[n-1]),
        "target_variance": target_variance, "target_reached": reached,
        "sample_count": int(x.shape[0]), "hidden_dim": int(x.shape[1]),
    }


def layer_numbers(spec, number_of_layers):
    if spec == "all":
        return list(range(1, number_of_layers + 1))
    layers = sorted({int(s) for s in spec.replace(",", " ").split()})
    if not layers or not all(1 <= x <= number_of_layers for x in layers):
        raise ValueError(f"Layer numbers must be 1..{number_of_layers}")
    return layers


def as_ppl_ratio(intervened_nll, baseline_nll):
    return math.exp(float(intervened_nll) - float(baseline_nll))
