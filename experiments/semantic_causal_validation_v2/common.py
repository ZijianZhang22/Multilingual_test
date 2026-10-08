#!/usr/bin/env python3
"""Shared utilities for semantic causal validation v2."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.random_layer_freeze_pilot.run_pilot import get_layers, load_model  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def replace_hidden(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden, *output[1:])
    return new_hidden


def _recursive_find_tensor(obj, key: str):
    if isinstance(obj, dict):
        if key in obj and torch.is_tensor(obj[key]):
            return obj[key]
        for value in obj.values():
            found = _recursive_find_tensor(value, key)
            if found is not None:
                return found
    return None


def load_basis(path: str, key: str) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu")
    q = None
    if isinstance(payload, dict) and "subspaces" in payload and key in payload["subspaces"]:
        q = payload["subspaces"][key]
    elif isinstance(payload, dict) and "anchor_subspaces" in payload and key in payload["anchor_subspaces"]:
        q = payload["anchor_subspaces"][key]
    else:
        q = _recursive_find_tensor(payload, key)

    if q is None:
        available = []
        if isinstance(payload, dict):
            available = list(payload.keys())
            if "subspaces" in payload and isinstance(payload["subspaces"], dict):
                available += [f"subspaces.{k}" for k in payload["subspaces"]]
        raise KeyError(f"Could not find basis {key!r} in {path}. Candidates: {available[:50]}")

    q = q.float()
    q, _ = torch.linalg.qr(q, mode="reduced")

    center = None
    if isinstance(payload, dict) and torch.is_tensor(payload.get("center")):
        center = payload["center"].float()

    meta = {
        "path": str(path),
        "key": key,
        "rank": int(q.shape[1]),
        "artifact_layer": int(payload["layer"]) if isinstance(payload, dict) and "layer" in payload else None,
    }
    return q, center, meta


def parse_basis_specs(specs: Iterable[str], default_file: Optional[str] = None):
    out = {}
    for spec in specs:
        if "=" in spec:
            name, rhs = spec.split("=", 1)
        else:
            name, rhs = spec, spec
        if ":" in rhs:
            path, key = rhs.rsplit(":", 1)
        else:
            if default_file is None:
                raise ValueError(f"Basis spec {spec!r} needs path:key or --basis_file")
            path, key = default_file, rhs
        q, center, meta = load_basis(path, key)
        out[name] = {"q": q, "center": center, "meta": meta}
    return out


def load_xnli_rows(languages: List[str], split: str, n_per_lang: int, seed: int):
    rows = []
    for lang in languages:
        ds = load_dataset("facebook/xnli", lang, split=split)
        indices = [i for i in range(len(ds)) if int(ds[i]["label"]) in (0, 1, 2)]
        rng = random.Random(seed + sum(ord(c) for c in lang))
        rng.shuffle(indices)
        for idx in indices[:n_per_lang]:
            ex = ds[idx]
            rows.append({
                "example_id": f"{lang}:{split}:{idx}",
                "pair_id": f"{split}:{idx}",
                "language": lang,
                "label": int(ex["label"]),
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
            })
    return rows


def load_aligned_xnli(languages: List[str], split: str, n_pairs: int, seed: int):
    dsets = {lang: load_dataset("facebook/xnli", lang, split=split) for lang in languages}
    n = min(len(ds) for ds in dsets.values())
    ids = list(range(n))
    random.Random(seed).shuffle(ids)
    rows = []
    kept = 0
    for idx in ids:
        labels = [int(dsets[lang][idx]["label"]) for lang in languages]
        if len(set(labels)) != 1 or labels[0] not in (0, 1, 2):
            continue
        for lang in languages:
            ex = dsets[lang][idx]
            rows.append({
                "example_id": f"{lang}:{split}:{idx}",
                "pair_id": f"{split}:{idx}",
                "language": lang,
                "label": labels[0],
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
            })
        kept += 1
        if kept >= n_pairs:
            break
    return rows


def nli_prompt(row: dict) -> str:
    return (
        "Classify the relation between the premise and hypothesis. "
        "Answer with A, B, or C only.\n"
        "A = entailment\nB = neutral\nC = contradiction\n"
        f"Premise: {row['premise']}\n"
        f"Hypothesis: {row['hypothesis']}\nAnswer:"
    )


def language_prompt(row: dict, languages: List[str]) -> str:
    names = {"en": "English", "zh": "Chinese", "fr": "French", "de": "German", "es": "Spanish"}
    letters = [chr(ord("A") + i) for i in range(len(languages))]
    choices = "\n".join(f"{a} = {names.get(l, l)}" for a, l in zip(letters, languages))
    text = f"{row['premise']} {row['hypothesis']}"
    return (
        "Identify the language of the following text. "
        f"Answer with one of {', '.join(letters)} only.\n"
        f"{choices}\nText: {text}\nAnswer:"
    )


def choice_token_ids(tokenizer, letters: List[str]) -> List[int]:
    ids = []
    for letter in letters:
        cand = tokenizer.encode(" " + letter, add_special_tokens=False)
        if not cand:
            cand = tokenizer.encode(letter, add_special_tokens=False)
        if len(cand) != 1:
            raise ValueError(f"Choice {letter!r} is not a single token: {cand}")
        ids.append(cand[0])
    return ids


@torch.no_grad()
def forward_choice(model, tokenizer, prompt, choice_ids, *, device, use_bf16, hook=None):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    handle = hook() if hook is not None else None
    try:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and use_bf16),
        ):
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[0, -1, choice_ids].float()
        return torch.softmax(logits, dim=-1).detach().cpu()
    finally:
        if handle is not None:
            handle.remove()


@torch.no_grad()
def capture_layer_state(model, tokenizer, prompt, *, layer_no, device, use_bf16):
    layer = get_layers(model)[layer_no - 1]
    box = {}

    def fhook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        box["h"] = h.detach().float()

    handle = layer.register_forward_hook(fhook)
    try:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"].to(device)
        mask = enc.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and use_bf16),
        ):
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        handle.remove()

    h = box["h"][0]
    if mask is None:
        mean = h.mean(dim=0)
    else:
        m = mask[0].float().unsqueeze(-1)
        mean = (h * m).sum(dim=0) / m.sum().clamp_min(1.0)
    return {"last": h[-1].cpu(), "mean": mean.cpu()}


def build_last_token_ablation_hook(model, layer_no, q, center, norm_fraction):
    layer = get_layers(model)[layer_no - 1]
    device = next(model.parameters()).device
    q = q.to(device=device, dtype=torch.float32)
    c = None if center is None else center.to(device=device, dtype=torch.float32)

    def register():
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            hf = h.float().clone()
            x = hf[:, -1, :]
            xc = x if c is None else x - c
            p = (xc @ q) @ q.T
            target = float(norm_fraction) * xc.norm(dim=-1, keepdim=True)
            perturb = -p * (target / p.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            hf[:, -1, :] = x + perturb
            return replace_hidden(output, hf.to(dtype=h.dtype))
        return layer.register_forward_hook(hook)
    return register


def build_mean_shift_hook(model, layer_no, delta_vec):
    layer = get_layers(model)[layer_no - 1]
    device = next(model.parameters()).device
    delta = delta_vec.to(device=device, dtype=torch.float32)

    def register():
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            return replace_hidden(output, (h.float() + delta.view(1, 1, -1)).to(dtype=h.dtype))
        return layer.register_forward_hook(hook)
    return register


def save_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
