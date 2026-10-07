#!/usr/bin/env python3
"""Measure old/new language activation-gradient subspaces by layer.

For each requested layer we collect gradients dL/dh_l over several old-language
and new-language minibatches, flatten token gradients into d-dimensional samples,
and form covariance/Gram matrices. We then report:
  - top-r old/new gradient subspaces,
  - principal-angle/projection overlap,
  - gradient energy of new-language gradients inside old-language subspace,
  - gradient energy of old-language gradients inside new-language subspace,
  - effective dimensions / spectra.

This is a diagnostic only; it does not modify training.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.random_layer_freeze_pilot.run_pilot import (  # noqa: E402
    get_layers,
    load_model,
)
from invariance.train_sequence import load_blocks, make_loader  # noqa: E402


def top_basis_from_gram(gram, rank):
    evals, evecs = torch.linalg.eigh(gram.double())
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0).float()
    evecs = evecs[:, order].float()
    r = min(rank, evecs.shape[1])
    return evecs[:, :r], evals


def spectral_metrics(evals):
    total = evals.sum().clamp_min(1e-30)
    p = evals / total
    pr = (total * total) / evals.pow(2).sum().clamp_min(1e-30)
    entropy = -(p[p > 0] * p[p > 0].log()).sum()
    return {
        "participation_ratio": float(pr),
        "entropy_effective_rank": float(entropy.exp()),
        "top1_fraction": float(p[:1].sum()),
        "top8_fraction": float(p[:8].sum()),
        "top32_fraction": float(p[:32].sum()),
        "top64_fraction": float(p[:64].sum()),
    }


def subspace_stats(qa, qb):
    s = torch.linalg.svdvals(qa.T @ qb).clamp(0, 1)
    r = min(qa.shape[1], qb.shape[1])
    angles = torch.rad2deg(torch.acos(s))
    return {
        "projection_overlap": float(s.pow(2).sum() / max(r, 1)),
        "mean_angle_deg": float(angles.mean()),
        "max_cosine": float(s.max()),
        "min_cosine": float(s.min()),
    }


def collect_gradient_gram(
    checkpoint,
    blocks,
    *,
    layers,
    batch_size,
    max_batches,
    token_stride,
    device,
    use_bf16,
):
    model = load_model(checkpoint, device, use_bf16)
    model.train()
    # Keep parameter requires_grad enabled so the forward activations carry an
    # autograd graph. We never step an optimizer; parameter gradients are
    # discarded after each diagnostic batch.
    transformer_layers = get_layers(model)
    stats = {}
    handles = []
    latest = {}

    for layer_no in layers:
        if not 1 <= layer_no <= len(transformer_layers):
            raise ValueError(f"Invalid layer {layer_no}")
        stats[layer_no] = {
            "gram": None,
            "energy": 0.0,
            "n_vectors": 0,
        }

        def make_hook(lno):
            def hook(_module, _inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                h.retain_grad()
                latest[lno] = h
            return hook

        handles.append(
            transformer_layers[layer_no - 1].register_forward_hook(
                make_hook(layer_no)
            )
        )

    try:
        for bi, (x,) in enumerate(make_loader(blocks, batch_size)):
            if bi >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            latest.clear()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                out = model(input_ids=x, labels=x, use_cache=False)
                loss = out.loss
            loss.backward()

            for layer_no in layers:
                g = latest[layer_no].grad
                if g is None:
                    raise RuntimeError(f"No activation gradient at layer {layer_no}")
                gf = g.float()
                gf = gf[:, ::token_stride, :].reshape(-1, gf.shape[-1])
                gram = gf.T @ gf
                gram_cpu = gram.detach().cpu()
                if stats[layer_no]["gram"] is None:
                    stats[layer_no]["gram"] = gram_cpu
                else:
                    stats[layer_no]["gram"].add_(gram_cpu)
                stats[layer_no]["energy"] += float(gf.pow(2).sum())
                stats[layer_no]["n_vectors"] += gf.shape[0]

            if (bi + 1) % 4 == 0:
                print(f"  gradient batches={bi+1}", flush=True)

    finally:
        for h in handles:
            h.remove()
        del model
        torch.cuda.empty_cache()

    return stats


def projected_energy(gram, q):
    num = torch.trace(q.T.double() @ gram.double() @ q.double())
    den = torch.trace(gram.double()).clamp_min(1e-30)
    return float(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="mechanism_runs/step1_layer_lambda_sweep/checkpoints/full_ft",
    )
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--layers", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24])
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_batches", type=int, default=16)
    ap.add_argument("--token_stride", type=int, default=8)
    ap.add_argument(
        "--out_dir",
        default="mechanism_runs/advanced_functional_diagnostics_v1/gradient_subspaces",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    old_blocks = load_blocks(Path(args.data_dir) / f"{args.old_language}_val.pt")
    new_blocks = load_blocks(Path(args.data_dir) / f"{args.new_language}_val.pt")

    print("Collecting old-language activation gradients...")
    old = collect_gradient_gram(
        args.checkpoint,
        old_blocks,
        layers=args.layers,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        token_stride=args.token_stride,
        device=device,
        use_bf16=use_bf16,
    )

    print("Collecting new-language activation gradients...")
    new = collect_gradient_gram(
        args.checkpoint,
        new_blocks,
        layers=args.layers,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        token_stride=args.token_stride,
        device=device,
        use_bf16=use_bf16,
    )

    rows = []
    artifact = {}
    for layer in args.layers:
        q_old, evals_old = top_basis_from_gram(old[layer]["gram"], args.rank)
        q_new, evals_new = top_basis_from_gram(new[layer]["gram"], args.rank)
        ov = subspace_stats(q_old, q_new)

        row = {
            "layer": layer,
            "rank": int(q_old.shape[1]),
            "old_gradient_energy": old[layer]["energy"],
            "new_gradient_energy": new[layer]["energy"],
            "new_energy_in_old_subspace": projected_energy(
                new[layer]["gram"], q_old
            ),
            "old_energy_in_new_subspace": projected_energy(
                old[layer]["gram"], q_new
            ),
            "old_new_projection_overlap": ov["projection_overlap"],
            "old_new_mean_principal_angle_deg": ov["mean_angle_deg"],
            "old_new_max_cosine": ov["max_cosine"],
            "old_new_min_cosine": ov["min_cosine"],
        }
        for k, v in spectral_metrics(evals_old).items():
            row[f"old_{k}"] = v
        for k, v in spectral_metrics(evals_new).items():
            row[f"new_{k}"] = v
        rows.append(row)

        artifact[str(layer)] = {
            "old_basis": q_old,
            "new_basis": q_new,
            "old_eigenvalues": evals_old,
            "new_eigenvalues": evals_new,
            "old_gram": old[layer]["gram"],
            "new_gram": new[layer]["gram"],
        }

        print(
            f"layer={layer:>2} overlap={row['old_new_projection_overlap']:.4f} "
            f"new->old={row['new_energy_in_old_subspace']:.4f} "
            f"old->new={row['old_energy_in_new_subspace']:.4f}"
        )

    with (out / "gradient_subspace_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    torch.save(
        {
            "checkpoint": args.checkpoint,
            "old_language": args.old_language,
            "new_language": args.new_language,
            "rank": args.rank,
            "layers": artifact,
        },
        out / "gradient_subspaces.pt",
    )
    (out / "manifest.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    print(f"Saved: {out / 'gradient_subspace_metrics.csv'}")
    print(f"Saved: {out / 'gradient_subspaces.pt'}")


if __name__ == "__main__":
    main()
