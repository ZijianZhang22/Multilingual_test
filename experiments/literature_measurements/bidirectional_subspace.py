#!/usr/bin/env python3
"""Bidirectional same-input Anchor/Adapted subspace interventions on LM loss.

Restoration: adapted h + alpha P(h_anchor - h_adapted).
Induction:   anchor h  + alpha P(h_adapted - h_anchor).
Rank-matched random bases use per-batch, per-rank SHRINK-ONLY equal norms.
This tests functional reversibility, not the historical origin of forgetting.
"""
import argparse
import json
from pathlib import Path

import torch

from experiments.literature_measurements.core import (
    get_layers, hidden_from_output, load_model, save_csv, with_hidden
)
from experiments.literature_measurements.subspace_core import REAL_SPACES, get_energy_scales, load_core, projected
from invariance.train_sequence import evaluate, load_blocks, make_loader


@torch.no_grad()
def eval_direction(anchor, adapted, blocks, batch_size, device, layer, spaces,
                   alpha, energy_mode, direction, bf16):
    source, target = (anchor, adapted) if direction == "restore" else (adapted, anchor)
    source.eval()
    target.eval()
    holder = {}
    sums = {name: {"loss": 0., "tokens": 0, "energy": 0., "natural": 0., "count": 0}
            for name in spaces}
    hs = get_layers(source)[layer-1].register_forward_hook(
        lambda _m, _i, output: holder.update(h=hidden_from_output(output).detach()))
    try:
        for (x,) in make_loader(blocks, batch_size):
            x = x.to(device)
            holder.clear()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=bf16 and str(device).startswith("cuda")):
                source(input_ids=x, use_cache=False)
                assert "h" in holder
                source_hidden = holder["h"]
                captured = {}

                def capture(_module, _inputs, output):
                    captured["h"] = hidden_from_output(output).detach()
                cap = get_layers(target)[layer-1].register_forward_hook(capture)
                try:
                    target(input_ids=x, use_cache=False)
                finally:
                    cap.remove()
                htarget = captured["h"]
                if htarget.shape != source_hidden.shape:
                    raise ValueError("Anchor/Adapted states not aligned")
                delta = source_hidden.float() - htarget.float()
                pieces = {(name, q.shape[1]): projected(delta, q)
                          for name, q in spaces.items()}
                scales, norms = get_energy_scales(pieces, mode=energy_mode)
                for (name, rank), part in pieces.items():
                    change = alpha * scales[(name, rank)] * part
                    def patch(_module, _inputs, output, perturb=change):
                        h = hidden_from_output(output)
                        return with_hidden(output, (h.float() + perturb).to(h.dtype))
                    handle = get_layers(target)[layer-1].register_forward_hook(patch)
                    try:
                        out = target(input_ids=x, labels=x, use_cache=False)
                    finally:
                        handle.remove()
                    n = x.shape[0] * (x.shape[1] - 1)
                    stats = sums[name]
                    stats["loss"] += float(out.loss) * n
                    stats["tokens"] += n
                    stats["energy"] += float(change.square().sum())
                    stats["natural"] += norms[(name, rank)] ** 2
                    stats["count"] += change.numel()
    finally:
        hs.remove()
    return {name: {
        "nll": s["loss"]/s["tokens"],
        "mean_delta_rms": (s["energy"]/s["count"]) ** .5,
        "natural_delta_rms": (s["natural"]/s["count"]) ** .5,
        "tokens": s["tokens"]
    } for name, s in sums.items() if s["tokens"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor_checkpoint", required=True)
    ap.add_argument("--adapted_checkpoint", required=True)
    ap.add_argument("--core_file", required=True)
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--spaces", nargs="+", choices=REAL_SPACES, default=list(REAL_SPACES))
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--alpha", type=float, default=1.)
    ap.add_argument("--energy_mode", choices=["matched","raw"], default="matched")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_blocks", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--allow_legacy_core", action="store_true")
    a = ap.parse_args()
    if not (0 < a.alpha <= 1):
        ap.error("--alpha must be in (0,1]")
    if a.old_language == a.new_language:
        ap.error("Languages must differ")
    core, spaces = load_core(a.core_file, a.spaces,
                             allow_legacy=a.allow_legacy_core)
    layer = int(core["layer"])
    device = torch.device(a.device)
    bf16 = (not a.no_bf16 and device.type == "cuda"
            and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    anchor, _ = load_model(a.anchor_checkpoint, device, bf16=bf16)
    adapted, _ = load_model(a.adapted_checkpoint, device, bf16=bf16)
    for model in (anchor, adapted):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    if (layer > len(get_layers(anchor)) or layer > len(get_layers(adapted))
            or anchor.config.hidden_size != adapted.config.hidden_size
            or anchor.config.hidden_size != int(core["hidden_dim"])):
        raise ValueError("Incompatible checkpoints or layer/subspace dimensionality")
    data = {}
    base = {}
    for lang in (a.old_language, a.new_language):
        blocks = load_blocks(Path(a.data_dir)/f"{lang}_val.pt")
        if a.max_blocks > 0:
            blocks = blocks[:a.max_blocks]
        data[lang] = blocks
        base[lang] = {
            "anchor": evaluate(anchor, blocks, a.batch_size, device, bf16),
            "adapted": evaluate(adapted, blocks, a.batch_size, device, bf16)
        }
    # invariance.train_sequence.evaluate() switches models back to train mode.
    # Disable dropout before all causal probes and baseline comparisons.
    anchor.eval()
    adapted.eval()
    results = []
    for lang, blocks in data.items():
        for direction in ("restore", "induce"):
            outcomes = eval_direction(anchor, adapted, blocks, a.batch_size, device,
                                      layer, spaces, a.alpha, a.energy_mode, direction, bf16)
            target_key = "adapted" if direction == "restore" else "anchor"
            base_loss = base[lang][target_key]
            for name, stats in outcomes.items():
                old_gap = base[lang]["adapted"] - base[lang]["anchor"]
                results.append({
                    "layer": layer, "language": lang, "direction": direction,
                    "subspace": name, "rank": spaces[name].shape[1],
                    "space_type": "random" if name.startswith("random_") else "real",
                    "alpha": a.alpha, "energy_mode": a.energy_mode,
                    "anchor_loss": base[lang]["anchor"],
                    "adapted_loss": base[lang]["adapted"],
                    "target_baseline_loss": base_loss,
                    "intervention_loss": stats["nll"],
                    "loss_delta": stats["nll"] - base_loss,
                    "recovery_fraction": ((base_loss - stats["nll"])/old_gap
                        if direction=="restore" and lang==a.old_language and old_gap > 1e-8
                        else ""),
                    "induced_forgetting_fraction": ((stats["nll"]-base_loss)/old_gap
                        if direction=="induce" and lang==a.old_language and old_gap > 1e-8
                        else ""),
                    "delta_rms": stats["mean_delta_rms"],
                    "natural_delta_rms": stats["natural_delta_rms"],
                    "eval_tokens": stats["tokens"]
                })
                print(f"[{direction}] {lang} {name}: loss={stats['nll']:.6f}", flush=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out/"bidirectional_subspace_summary.csv", results)
    (out/"bidirectional_protocol.json").write_text(json.dumps({
        **vars(a), "core_fit_protocol": core.get("fit_protocol", "legacy"),
        "layer": layer, "hook_site": "decoder block outputs (all tokens)",
        "matched_energy": "batch/rank minimum across real and random; shrink-only",
        "interpretation": "causal reversibility of a representation intervention, not training-origin proof"
    }, indent=2))
    print(f"Saved {len(results)} comparisons to {out}")


if __name__ == "__main__":
    main()
