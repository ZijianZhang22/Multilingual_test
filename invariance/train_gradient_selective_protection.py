import argparse
import csv
import json
from pathlib import Path

import torch

from advanced_preservation_utils import (
    estimate_hidden_gradient_importance,
    evaluate,
    fit_hidden_gradient_subspace,
    hidden_gradient_projection_hook,
    load_blocks,
    load_model,
    make_loader,
    make_stability_weights,
    set_seed,
    shuffled_blocks,
)


def coordinate_gradient_hook(weights, strength):
    # Convert positive stability scores to [0, 1].
    w = weights.float()
    w = w / w.max().clamp_min(1e-12)
    scale = 1.0 - strength * w
    scale = scale.clamp_min(0.0)

    def hook(grad):
        return (grad.float() * scale.view(1, 1, -1)).to(grad.dtype)

    return hook


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Method 2: gradient-aware selective protection. Modify the gradient "
            "flow at a multilingual hidden layer instead of adding an L2 penalty."
        )
    )
    ap.add_argument("--anchor_checkpoint", required=True)
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--eval_languages", nargs="+", default=["en", "zh", "fr"])
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument(
        "--surgery_mode",
        choices=["coordinate", "subspace"],
        default="coordinate",
        help=(
            "coordinate: suppress hidden gradients using old/new Fisher-like "
            "importance per coordinate; subspace: project gradients away from "
            "top old-language hidden-gradient covariance directions."
        ),
    )
    ap.add_argument("--strength", type=float, default=0.75)
    ap.add_argument("--subspace_rank", type=int, default=64)
    ap.add_argument(
        "--importance_mode",
        choices=["ratio", "old_fraction", "old_only"],
        default="old_fraction",
    )
    ap.add_argument("--importance_batches", type=int, default=8)
    ap.add_argument("--importance_batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_shuffle_seed", type=int, default=1001)
    ap.add_argument("--max_train_blocks", type=int, default=None)
    ap.add_argument(
        "--out_dir",
        default="invariance_runs/advanced_methods/gradient_selective",
    )
    ap.add_argument("--no_bf16", action="store_true")
    args = ap.parse_args()

    if not 0.0 <= args.strength <= 1.0:
        raise ValueError("--strength must be between 0 and 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    use_bf16 = (not args.no_bf16) and torch.cuda.is_bf16_supported()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    old_train = load_blocks(Path(args.data_dir) / f"{args.old_language}_train.pt")
    new_train_raw = load_blocks(Path(args.data_dir) / f"{args.new_language}_train.pt")
    new_train = shuffled_blocks(
        new_train_raw, args.data_shuffle_seed, args.max_train_blocks
    )
    val = {
        lang: load_blocks(Path(args.data_dir) / f"{lang}_val.pt")
        for lang in args.eval_languages
    }

    # Evaluate the anchor before freeing it. Training then uses only one model.
    anchor = load_model(
        args.anchor_checkpoint, device, use_bf16, trainable=False
    )
    anchor_losses = {
        lang: evaluate(anchor, val[lang], args.eval_batch, device, use_bf16)
        for lang in args.eval_languages
    }
    del anchor
    torch.cuda.empty_cache()

    artifact = {
        "layer": args.layer,
        "surgery_mode": args.surgery_mode,
        "strength": args.strength,
    }

    if args.surgery_mode == "coordinate":
        print("Estimating old hidden-gradient importance...")
        old_imp = estimate_hidden_gradient_importance(
            args.anchor_checkpoint,
            old_train,
            layer=args.layer,
            batch_size=args.importance_batch,
            max_batches=args.importance_batches,
            device=device,
            use_bf16=use_bf16,
        )
        print("Estimating new hidden-gradient importance...")
        new_imp = estimate_hidden_gradient_importance(
            args.anchor_checkpoint,
            new_train_raw,
            layer=args.layer,
            batch_size=args.importance_batch,
            max_batches=args.importance_batches,
            device=device,
            use_bf16=use_bf16,
        )
        weights = make_stability_weights(
            old_imp, new_imp, mode=args.importance_mode
        )
        hook_factory = lambda: coordinate_gradient_hook(
            weights.to(device), args.strength
        )
        artifact.update({
            "old_importance": old_imp,
            "new_importance": new_imp,
            "stability_weights": weights,
            "importance_mode": args.importance_mode,
        })
    else:
        print("Fitting old-language hidden-gradient protection subspace...")
        q_old, evals = fit_hidden_gradient_subspace(
            args.anchor_checkpoint,
            old_train,
            layer=args.layer,
            rank=args.subspace_rank,
            batch_size=args.importance_batch,
            max_batches=args.importance_batches,
            device=device,
            use_bf16=use_bf16,
        )
        q_old = q_old.to(device)
        hook_factory = lambda: hidden_gradient_projection_hook(
            q_old, args.strength
        )
        artifact.update({
            "protected_subspace_basis": q_old.cpu(),
            "protected_subspace_eigenvalues": evals,
            "subspace_rank": int(q_old.shape[1]),
        })

    torch.save(artifact, out_dir / "gradient_protection_artifact.pt")

    model = load_model(
        args.anchor_checkpoint, device, use_bf16, trainable=True
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    opt.zero_grad(set_to_none=True)

    loader = make_loader(new_train, args.micro_batch)
    total_lm = 0.0
    n_batches = 0
    tokens_seen = 0

    for i, (x,) in enumerate(loader):
        x = x.to(device, non_blocking=True)
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
        h = out.hidden_states[args.layer]
        handle = h.register_hook(hook_factory())

        (out.loss / args.grad_accum).backward()
        handle.remove()

        total_lm += float(out.loss.detach())
        n_batches += 1
        tokens_seen += x.numel()

        do_step = ((i + 1) % args.grad_accum == 0) or (i + 1 == len(loader))
        if do_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if (i + 1) % 50 == 0:
            print(
                f"batch={i+1}/{len(loader)} lm={total_lm/n_batches:.4f}"
            )

    final_losses = {
        lang: evaluate(model, val[lang], args.eval_batch, device, use_bf16)
        for lang in args.eval_languages
    }

    row = {
        "method": f"gradient_selective_{args.surgery_mode}",
        "seed": args.seed,
        "layer": args.layer,
        "strength": args.strength,
        "old_language": args.old_language,
        "new_language": args.new_language,
        "tokens_seen": tokens_seen,
        "mean_train_lm_loss": total_lm / max(n_batches, 1),
        "forgetting_loss_delta": (
            final_losses[args.old_language] - anchor_losses[args.old_language]
        ),
        "new_language_gain": (
            anchor_losses[args.new_language] - final_losses[args.new_language]
        ),
    }
    for lang in args.eval_languages:
        row[f"{lang}_anchor_loss"] = anchor_losses[lang]
        row[f"{lang}_final_loss"] = final_losses[lang]
        row[f"{lang}_loss_delta"] = final_losses[lang] - anchor_losses[lang]

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    (out_dir / "manifest.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    print("\n=== Gradient-aware Selective Protection ===")
    print(
        f"mode={args.surgery_mode} strength={args.strength:g} "
        f"forget={row['forgetting_loss_delta']:+.6f} "
        f"new_gain={row['new_language_gain']:+.6f}"
    )
    for lang in args.eval_languages:
        print(
            f"{lang}: {anchor_losses[lang]:.4f} -> {final_losses[lang]:.4f} "
            f"delta={final_losses[lang]-anchor_losses[lang]:+.4f}"
        )
    print(f"Saved: {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
