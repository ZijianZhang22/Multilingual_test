import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from advanced_preservation_utils import load_blocks, load_model, make_loader, shuffled_blocks


def cosine(a, b, eps=1e-12):
    a = a.float()
    b = b.float()
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(eps))


def collect_hidden_gradients(
    model,
    blocks,
    layers,
    *,
    batch_size,
    max_batches,
    device,
    use_bf16,
):
    per_layer = {layer: [] for layer in layers}
    model.eval()
    for batch_idx, (x,) in enumerate(make_loader(blocks, batch_size)):
        if batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and use_bf16),
        ):
            out = model(
                input_ids=x,
                labels=x,
                output_hidden_states=True,
                use_cache=False,
            )
        hs = out.hidden_states
        chosen = []
        for layer in layers:
            idx = layer if layer >= 0 else len(hs) + layer
            if idx < 0 or idx >= len(hs):
                raise ValueError(
                    f"Requested layer {layer}; model returned {len(hs)} hidden states"
                )
            chosen.append(hs[idx])
        grads = torch.autograd.grad(out.loss, chosen, retain_graph=False)
        for layer, grad in zip(layers, grads):
            vec = grad.float().mean(dim=(0, 1)).detach().cpu()
            per_layer[layer].append(vec)
    return per_layer


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Measure OLD-vs-NEW hidden-gradient interference at a fixed anchor model. "
            "Negative cosine indicates locally conflicting functional update directions."
        )
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", required=True)
    ap.add_argument("--new_language", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24])
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_batches", type=int, default=8)
    ap.add_argument("--max_blocks", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--no_bf16", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    device = torch.device("cuda")
    use_bf16 = (not args.no_bf16) and torch.cuda.is_bf16_supported()

    old_path = Path(args.data_dir) / f"{args.old_language}_val.pt"
    new_path = Path(args.data_dir) / f"{args.new_language}_val.pt"
    if not old_path.exists() or not new_path.exists():
        raise FileNotFoundError(f"Need both {old_path} and {new_path}")

    old_blocks = shuffled_blocks(load_blocks(old_path), args.seed + 101, args.max_blocks)
    new_blocks = shuffled_blocks(load_blocks(new_path), args.seed + 202, args.max_blocks)

    model = load_model(args.checkpoint, device, use_bf16, trainable=True)
    old_g = collect_hidden_gradients(
        model,
        old_blocks,
        args.layers,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=device,
        use_bf16=use_bf16,
    )
    new_g = collect_hidden_gradients(
        model,
        new_blocks,
        args.layers,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=device,
        use_bf16=use_bf16,
    )
    del model
    torch.cuda.empty_cache()

    rows = []
    for layer in args.layers:
        old_vecs = old_g[layer]
        new_vecs = new_g[layer]
        n = min(len(old_vecs), len(new_vecs))
        if n == 0:
            continue
        batch_cos = np.array([cosine(old_vecs[i], new_vecs[i]) for i in range(n)])
        old_mean = torch.stack(old_vecs[:n]).mean(dim=0)
        new_mean = torch.stack(new_vecs[:n]).mean(dim=0)
        rows.append({
            "layer": layer,
            "n_batch_pairs": n,
            "mean_gradient_cosine": cosine(old_mean, new_mean),
            "batch_cosine_mean": float(batch_cos.mean()),
            "batch_cosine_std": float(batch_cos.std(ddof=1)) if n > 1 else 0.0,
            "batch_cosine_min": float(batch_cos.min()),
            "batch_cosine_max": float(batch_cos.max()),
            "negative_batch_fraction": float((batch_cos < 0).mean()),
            "old_gradient_norm": float(old_mean.norm().item()),
            "new_gradient_norm": float(new_mean.norm().item()),
        })

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== OLD vs NEW hidden-gradient interference ===")
    for r in rows:
        print(
            f"layer={r['layer']:>2} mean_cos={r['mean_gradient_cosine']:+.4f} "
            f"batch_mean={r['batch_cosine_mean']:+.4f} "
            f"neg_frac={r['negative_batch_fraction']:.2f}"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
