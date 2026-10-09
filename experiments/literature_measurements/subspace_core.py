"""Shared, deliberately narrow Drift / Transfer / ISR causal intervention helpers.

All interventions are on the OUTPUT of the same 1-based transformer block.
Only orthonormal linear directions are used. A fitted basis is NOT a semantic
mechanism by itself.
"""
from collections import defaultdict
from pathlib import Path
import json
import torch

from experiments.literature_measurements.core import orthonormal, hidden_from_output, with_hidden

REAL_SPACES = ("drift", "transfer", "isr_cov", "isr_multiclass")


def load_core(path, selected=REAL_SPACES, *, allow_legacy=False, include_random=True):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    protocol = str(payload.get("fit_protocol", ""))
    if not allow_legacy and not protocol.startswith("probe_train_only"):
        raise ValueError(
            "Basis provenance not certified train-only. Refit using updated "
            "build_core_subspaces.py, or explicitly pass --allow_legacy_core "
            "for exploratory (NOT held-out) analysis."
        )
    if not isinstance(payload.get("layer"), int):
        raise ValueError("Core payload requires 1-based integer layer")
    selection = [s.strip() for s in selected if s.strip()]
    if not selection or any(s not in REAL_SPACES for s in selection):
        raise ValueError(f"Supported subspaces only: {REAL_SPACES}")
    available = payload.get("subspaces", {})
    result = {}
    for name in selection:
        if name not in available:
            raise ValueError(f"Missing basis: {name}")
        basis = orthonormal(available[name])
        if basis.shape[0] != int(payload["hidden_dim"]):
            raise ValueError(f"Wrong hidden dimension for {name}")
        result[name] = basis
        if include_random:
            ctrl_name = payload.get("matched_random_controls", {}).get(name, "random_" + name)
            if ctrl_name in available:
                random_q = orthonormal(available[ctrl_name])
            else:
                rank = basis.shape[1]
                random_q = orthonormal(torch.randn(
                    basis.shape[0], rank,
                    generator=torch.Generator().manual_seed(4101 + REAL_SPACES.index(name))))
            if random_q.shape != basis.shape:
                raise ValueError(f"Random control must match rank of {name}")
            result["random_" + name] = random_q
    return payload, result


def projected(x, basis):
    q = basis.to(device=x.device, dtype=torch.float32)
    xx = x.float()
    return (xx @ q) @ q.T


def get_energy_scales(vectors, mode="matched", eps=1e-12):
    """Per-example or per-batch equal-norm SHRINK-only controls, grouped by rank.

    vectors: {(name, rank): projected_tensor}. All tensors in each group
    must be shape-matched. Target is minimum nonnegative magnitude across
    included real and random spaces; for zero projection every scale is 0.
    """
    if mode not in ("raw", "matched"):
        raise ValueError(mode)
    norms = {}
    for key, vector in vectors.items():
        norms[key] = vector.float().square().sum().sqrt().item()
    if mode == "raw":
        return {k: 1.0 for k in vectors}, norms
    by_rank = defaultdict(list)
    for (_, rank), norm in norms.items():
        by_rank[rank].append(norm)
    minima = {rank: min(values) for rank, values in by_rank.items()}
    scales = {key: (min(1.0, minima[key[1]]/norm) if norm > eps else 0.0)
              for key, norm in norms.items()}
    return scales, norms


def change_last_token(output, perturbation):
    h = hidden_from_output(output)
    if h.shape[0] != 1 or perturbation.shape != h[0,-1].shape:
        raise ValueError("Last-token patch needs batch=1 and matching hidden dim")
    patched = h.clone()
    patched[0,-1] = (h[0,-1].float() + perturbation).to(h.dtype)
    return with_hidden(output, patched)


def xent_for_options(logits, gold):
    lp = logits.float().log_softmax(-1)
    return {
        "accuracy": int(int(logits.argmax()) == int(gold)),
        "gold_nll": float(-lp[gold]), "gold_prob": float(lp[gold].exp())
    }


def classify_name(name):
    if name in REAL_SPACES:
        return "real"
    if name.startswith("random_") and name[7:] in REAL_SPACES:
        return "random"
    raise ValueError(f"Unknown/unsupported subspace: {name}")


def choose_donors(rows, target, seed=2026):
    """Choose label and language controls with no re-used target pair.

    Paired cross-language donor must be the same aligned XNLI item. Unrelated
    same-label donors are *NOT* assumed semantically equivalent.
    """
    import random
    rng = random.Random(seed)
    pid = str(target["pair_id"])
    lang, label = target["language"], int(target["label"])
    groups = {
        "same_pair_cross_lang": lambda r: str(r["pair_id"]) == pid and r["language"] != lang,
        "different_pair_same_lang_same_label": lambda r: str(r["pair_id"]) != pid and r["language"] == lang and int(r["label"]) == label,
        "different_pair_cross_lang_same_label": lambda r: str(r["pair_id"]) != pid and r["language"] != lang and int(r["label"]) == label,
        "different_pair_same_lang_diff_label": lambda r: str(r["pair_id"]) != pid and r["language"] == lang and int(r["label"]) != label,
        "different_pair_cross_lang_diff_label": lambda r: str(r["pair_id"]) != pid and r["language"] != lang and int(r["label"]) != label,
    }
    selected = {}
    for key, filt in groups.items():
        choices = [r for r in rows if filt(r) and r["example_id"] != target["example_id"]]
        if choices:
            selected[key] = rng.choice(sorted(choices, key=lambda x: str(x["example_id"])))
    return selected


def nli_prompt(row):
    return (
        "Read the premise and hypothesis. Choose one letter: "
        "A = entailment, B = neutral, C = contradiction.\n"
        f'Premise: {row["premise"]}\nHypothesis: {row["hypothesis"]}\nAnswer:'
    )


def candidate_ids(tokenizer):
    results = [tokenizer.encode(" " + letter, add_special_tokens=False)
               for letter in ("A", "B", "C")]
    if any(len(v) != 1 for v in results) or len({v[0] for v in results}) != 3:
        raise ValueError("Answer letters must have distinct single-token encodings")
    return [v[0] for v in results]
