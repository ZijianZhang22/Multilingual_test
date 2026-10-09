#!/usr/bin/env python3
"""Full-layer activation patching and optional whole-layer weight restoration.

Loads the same Stage-1 anchor and Stage-2 adapted checkpoints. Unlike
subspace-specific rescue, this performs full-state anchor -> adapted patching
without preselecting any linear representation direction.

Uses Wiki-style val tensors: {language}_val.pt containing input_ids.
"""
import argparse
from pathlib import Path

import torch

from experiments.literature_measurements.core import (
    get_layers, hidden_from_output, layer_numbers, save_csv, with_hidden
)
from experiments.random_layer_freeze_pilot.run_pilot import load_model
from invariance.train_sequence import evaluate, load_blocks


@torch.no_grad()
def full_activation_patch(anchor, adapted, blocks, batch_size, device, use_bf16, layer, alpha=1.0):
    from invariance.train_sequence import make_loader
    anchor.eval()
    adapted.eval()
    shared = {"h": None}

    def anchor_hook(_module, _inputs, out):
        shared["h"] = hidden_from_output(out).detach()

    def adapted_hook(_module, _inputs, out):
        h = hidden_from_output(out)
        if shared["h"] is None:
            raise RuntimeError("Anchor forward must run before adapted forward")
        if h.shape != shared["h"].shape:
            raise ValueError("Anchor/adapted activation shape mismatch")
        new = h.float() + alpha * (shared["h"].float() - h.float())
        return with_hidden(out, new.to(h.dtype))

    ha = get_layers(anchor)[layer-1].register_forward_hook(anchor_hook)
    hb = get_layers(adapted)[layer-1].register_forward_hook(adapted_hook)
    total = count = 0
    try:
        for (x,) in make_loader(blocks, batch_size):
            x = x.to(device)
            shared["h"] = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=str(device).startswith("cuda") and use_bf16):
                anchor(input_ids=x, use_cache=False)
                result = adapted(input_ids=x, labels=x, use_cache=False)
            n = x.shape[0] * (x.shape[1] - 1)
            total += float(result.loss) * n
            count += n
    finally:
        ha.remove()
        hb.remove()
    if count == 0:
        raise ValueError("No eval target tokens")
    return total/count


@torch.no_grad()
def weight_restore_evaluate(anchor, adapted, blocks, batch_size, device, use_bf16, layer):
    """Restore one adapted decoder layer with anchor weights; always undo."""
    ma = get_layers(anchor)[layer-1]
    mb = get_layers(adapted)[layer-1]
    before = {name: p.detach().clone() for name, p in mb.state_dict().items()}
    try:
        mb.load_state_dict(ma.state_dict(), strict=True)
        adapted.eval()
        loss = evaluate(adapted, blocks, batch_size, device, use_bf16)
    finally:
        mb.load_state_dict(before, strict=True)
    return loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor_checkpoint", required=True)
    p.add_argument("--adapted_checkpoint", required=True)
    p.add_argument("--data_dir", default="invariance_data/wiki")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--languages", nargs="+", default=["en", "zh"])
    p.add_argument("--old_language", default="en")
    p.add_argument("--layers", default="all")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_max_blocks", type=int, default=64)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--weight_restore", action="store_true",
                   help="Also run full-layer weight restoration (more costly)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()
    if not 0 < args.alpha <= 1:
        p.error("--alpha must be within (0, 1]")
    device = torch.device(args.device)
    use_bf16 = not args.no_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()

    anchor = load_model(args.anchor_checkpoint, device, use_bf16)
    adapted = load_model(args.adapted_checkpoint, device, use_bf16)
    for model in (anchor, adapted):
        model.eval()
        for par in model.parameters():
            par.requires_grad_(False)
    if len(get_layers(anchor)) != len(get_layers(adapted)):
        raise ValueError("Checkpoints have incompatible layer counts")
    layer_ids = layer_numbers(args.layers, len(get_layers(adapted)))
    val = {}
    for lang in args.languages:
        blocks = load_blocks(Path(args.data_dir) / f"{lang}_val.pt")
        if args.eval_max_blocks > 0:
            blocks = blocks[:args.eval_max_blocks]
        val[lang] = blocks
    baselines = {}
    for lang, blocks in val.items():
        baselines[lang] = (
            evaluate(anchor, blocks, args.batch_size, device, use_bf16),
            evaluate(adapted, blocks, args.batch_size, device, use_bf16),
        )
        print(f"{lang}: anchor={baselines[lang][0]:.6f}, "
              f"adapted={baselines[lang][1]:.6f}", flush=True)
    anchor.eval()
    adapted.eval()
    out = []
    for layer in layer_ids:
        for lang in args.languages:
            blocks = val[lang]
            base_a, base_b = baselines[lang]
            patched = full_activation_patch(anchor, adapted, blocks, args.batch_size,
                                             device, use_bf16, layer, alpha=args.alpha)
            wr = weight_restore_evaluate(anchor, adapted, blocks, args.batch_size,
                                         device, use_bf16, layer) if args.weight_restore else None
            gap = base_b - base_a
            out.append({
                "layer": layer, "language": lang, "alpha": args.alpha,
                "anchor_loss": base_a, "adapted_loss": base_b,
                "full_patch_loss": patched, "patch_delta_loss": patched-base_b,
                "patch_gap_recovery": (base_b-patched)/gap if abs(gap)>1e-9 else "",
                "weight_restore_loss": "" if wr is None else wr,
                "weight_restore_gap_recovery": (base_b-wr)/gap
                    if wr is not None and abs(gap)>1e-9 else "",
                "eval_tokens": len(blocks)*(blocks.shape[1]-1),
            })
            print(f"L{layer} {lang}: patched={patched:.6f}"
                  + (f" weight={wr:.6f}" if wr is not None else ""), flush=True)
    save_csv(args.out_csv, out)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
