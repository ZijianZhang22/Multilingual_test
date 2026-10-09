#!/usr/bin/env python3
"""Controlled, model-level semantic donor patching of Drift/Transfer/ISR bases.

Experimental evidence target: information use, NOT proof that the basis is a
pure semantic representation. Works on held-out *aligned XNLI test* items.

Every target is paired with:
 - same item, other language (translation, same NLI label)
 - different item, same language / other language, same label
 - different item, same language / other language, different label
The unrelated same-label examples control for cheap NLI label information.
Interventions change only the final prompt-token hidden state and use one
fitted basis and its rank-matched random basis, equalized within each donor.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from experiments.literature_measurements.core import (
    get_layers, load_model, rows_from_jsonl, save_csv
)
from experiments.literature_measurements.subspace_core import (
    REAL_SPACES, candidate_ids, change_last_token, choose_donors,
    classify_name, get_energy_scales, load_core, nli_prompt,
    projected, xent_for_options
)


def prepare_eval(path, languages, allow_other_split):
    rows = rows_from_jsonl(path, languages)
    for row in rows:
        if "pair_id" not in row or "example_id" not in row or "label" not in row:
            raise ValueError("Aligned eval data requires pair_id, example_id, label")
        if "premise" not in row or "hypothesis" not in row:
            raise ValueError("XNLI NLI prompts require premise and hypothesis")
        if not allow_other_split and not str(row["pair_id"]).startswith("test:"):
            raise ValueError(
                "Hold-out semantics requires XNLI --split test aligned pair IDs. "
                "Use --allow_other_split only for exploratory experiments."
            )
        if int(row["label"]) not in (0, 1, 2):
            raise ValueError("XNLI classes must be 0,1,2")
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["pair_id"])].append(row)
    for pair, members in groups.items():
        if len({int(r["label"]) for r in members}) != 1:
            raise ValueError(f"Conflicting labels for aligned pair {pair}")
        if len({r["language"] for r in members}) != len(members):
            raise ValueError(f"Multiple rows for same language in pair {pair}")
    return rows


def choose_targets(rows, languages, per_language, seed):
    rng = random.Random(seed)
    by_lang_label = defaultdict(list)
    for row in rows:
        by_lang_label[(row["language"], int(row["label"]))].append(row)
    targets = []
    for lang in languages:
        quota = [per_language // 3 + (int(y < per_language % 3)) for y in range(3)]
        for y, count in enumerate(quota):
            if not count:
                continue
            candidates = sorted(by_lang_label[(lang, y)], key=lambda v: v["example_id"])
            if len(candidates) < count:
                raise ValueError(f"Need {count} held-out examples for {lang} label {y}")
            targets.extend(rng.sample(candidates, count))
    rng.shuffle(targets)
    return targets


@torch.no_grad()
def run_forward(model, tokenizer, row, ids, device, layer, *, delta=None):
    inputs = tokenizer(nli_prompt(row), return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hook_state = {}

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if delta is None:
            hook_state["h"] = h[0, -1].float().detach().cpu().clone()
            return None
        return change_last_token(output, delta.to(device=h.device, dtype=torch.float32))

    handle = get_layers(model)[layer - 1].register_forward_hook(hook)
    try:
        out = model(**inputs, use_cache=False)
        options = out.logits[0, -1, ids].float().detach().cpu()
        return options, hook_state.get("h")
    finally:
        handle.remove()


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["layer"], row["subspace"], row["donor_condition"],
                row["energy_mode"], row["alpha"])].append(row)
    out = []
    for (layer, space, donor, energy_mode, alpha), arr in sorted(groups.items()):
        n = len(arr)
        def avg(key):
            values = [float(r[key]) for r in arr if r.get(key) is not None]
            return sum(values) / len(values) if values else ""
        out.append({
            "layer": layer, "subspace": space, "space_type": classify_name(space),
            "donor_condition": donor, "energy_mode": energy_mode,
            "alpha": alpha, "n": n, "n_correct_baseline": sum(int(r["baseline_accuracy"]) for r in arr),
            "gold_nll_delta": avg("gold_nll_delta"),
            "accuracy_delta": avg("accuracy_delta"),
            "donor_label_margin_delta": avg("donor_label_margin_delta"),
            "matched_energy_l2": avg("actual_delta_l2"),
            "natural_energy_l2": avg("natural_delta_l2"),
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Adapted checkpoint for model-level NLI patching")
    p.add_argument("--core_file", required=True, help="Fit-only core_subspaces.pt with pool=last")
    p.add_argument("--eval_jsonl", required=True, help="XNLI aligned test split, never fitted")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--spaces", nargs="+", default=list(REAL_SPACES),
                   choices=REAL_SPACES)
    p.add_argument("--languages", nargs="+", default=["en", "zh"])
    p.add_argument("--n_targets_per_language", type=int, default=6)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--energy_modes", nargs="+", choices=("matched", "raw"),
                   default=["matched"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--allow_other_split", action="store_true",
                   help="Non-test data are exploratory ONLY")
    p.add_argument("--allow_legacy_core", action="store_true",
                   help="Ignore train-only provenance check; results exploratory ONLY")
    a = p.parse_args()
    if a.alpha <= 0 or a.alpha > 1 or a.n_targets_per_language < 3:
        p.error("alpha must be (0,1] and target count >=3")

    core, spaces = load_core(a.core_file, a.spaces,
                             allow_legacy=a.allow_legacy_core)
    if core.get("pool") != "last":
        p.error("Must use a LAST-token fitted core; mean-pool is not interchangeable")
    model, tok = load_model(a.checkpoint, a.device, bf16=not a.no_bf16)
    layer = int(core["layer"])
    if layer > len(get_layers(model)):
        p.error("Core layer is outside model depth")
    if int(core["hidden_dim"]) != int(model.config.hidden_size):
        p.error("Subspace dimension/checkpoint mismatch")
    ids = candidate_ids(tok)
    rows = prepare_eval(a.eval_jsonl, a.languages, a.allow_other_split)
    targets = choose_targets(rows, a.languages, a.n_targets_per_language, a.seed)
    plans = []
    needed = {}
    for j, target in enumerate(targets):
        needed[str(target["example_id"])] = target
        donors = choose_donors(rows, target, a.seed + j)
        if "same_pair_cross_lang" not in donors:
            raise ValueError("No translated donor; check aligned languages")
        plans.append((target, donors))
        for donor in donors.values():
            needed[str(donor["example_id"])] = donor
    print(f"Semantic test: {len(targets)} targets, {len(needed)} unique prompts", flush=True)

    baseline = {}
    for j, (rid, row) in enumerate(sorted(needed.items())):
        tokens = tok(nli_prompt(row), add_special_tokens=True)["input_ids"]
        if len(tokens) > a.max_length:
            raise ValueError(f"Prompt too long ({len(tokens)}): {rid}")
        logits, h = run_forward(model, tok, row, ids, a.device, layer)
        baseline[rid] = (logits, h)
        if j % 20 == 0:
            print(f"Encoded {j+1}/{len(needed)}", flush=True)
    first_target = targets[0]
    original, h = baseline[str(first_target["example_id"])]
    zero, _ = run_forward(model, tok, first_target, ids, a.device, layer,
                          delta=torch.zeros_like(h))
    if float((zero - original).abs().max()) > 0.05:
        raise RuntimeError("Zero intervention changed model output")
    print(f"Zero-hook max logits difference: {float((zero-original).abs().max()):.4g}")

    outputs = []
    for j, (target, donors) in enumerate(plans):
        tid = str(target["example_id"])
        baseline_logits, h_target = baseline[tid]
        gold = int(target["label"])
        base_metrics = xent_for_options(baseline_logits, gold)
        lp_base = baseline_logits.log_softmax(-1)
        for condition, donor in donors.items():
            did = str(donor["example_id"])
            h_donor = baseline[did][1]
            displacement = h_donor - h_target
            pieces = {(name, q.shape[1]): projected(displacement, q)
                      for name, q in spaces.items()}
            for energy_mode in a.energy_modes:
                scales, norms = get_energy_scales(pieces, mode=energy_mode)
                for (space, rank), projected_delta in pieces.items():
                    natural = norms[(space, rank)]
                    scale = scales[(space, rank)]
                    change = a.alpha * scale * projected_delta
                    logits, _ = run_forward(model, tok, target, ids, a.device, layer,
                                            delta=change)
                    met = xent_for_options(logits, gold)
                    donor_label = int(donor["label"])
                    margin = None
                    if donor_label != gold:
                        margin = float((logits[donor_label] - logits[gold])
                                       - (baseline_logits[donor_label] - baseline_logits[gold]))
                    outputs.append({
                        "layer": layer, "subspace": space, "rank": rank,
                        "space_type": classify_name(space), "target_id": tid,
                        "donor_id": did, "target_language": target["language"],
                        "donor_language": donor["language"],
                        "target_label": gold, "donor_label": donor_label,
                        "donor_condition": condition, "same_pair":
                            int(str(target["pair_id"]) == str(donor["pair_id"])),
                        "same_label": int(gold == donor_label),
                        "energy_mode": energy_mode, "alpha": a.alpha,
                        "baseline_accuracy": base_metrics["accuracy"],
                        "patched_accuracy": met["accuracy"],
                        "accuracy_delta": met["accuracy"] - base_metrics["accuracy"],
                        "baseline_gold_nll": base_metrics["gold_nll"],
                        "patched_gold_nll": met["gold_nll"],
                        "gold_nll_delta": met["gold_nll"] - base_metrics["gold_nll"],
                        "donor_label_margin_delta": margin,
                        "natural_delta_l2": natural,
                        "actual_delta_l2": float(change.norm()),
                        "scale_shrink_only": scale,
                        "baseline_correct_prob": float(lp_base[gold].exp()),
                        "patched_correct_prob": met["gold_prob"],
                    })
        print(f"Target {j+1}/{len(plans)}: {tid}", flush=True)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "semantic_patching_examples.csv", outputs)
    save_csv(out / "semantic_patching_summary.csv", summarize(outputs))
    protocol = {
        **vars(a), "layer": layer, "fit_protocol": core.get("fit_protocol", "legacy"),
        "intervention_site": "decoder block output, last prompt token",
        "donor_design": "same aligned translation vs unrelated same/different language and label",
        "matched_energy": "within each donor and rank, shrink-only to minimum projected norm",
        "interpretation": "NLI label-level functional sensitivity, NOT proof of full semantics",
        "limitation": "Subspaces fitted on unprompted texts; patching uses prompted NLI"
    }
    (out / "semantic_protocol.json").write_text(json.dumps(protocol, indent=2))
    print(f"Saved {len(outputs)} interventions to {out}")


if __name__ == "__main__":
    main()
