import argparse
import csv
import json
from pathlib import Path

import torch

from advanced_preservation_utils import (
    estimate_sae_feature_importance,
    evaluate,
    load_blocks,
    load_model,
    make_loader,
    sample_hidden_tokens,
    select_top_feature_mask,
    set_seed,
    shuffled_blocks,
    train_sparse_autoencoder,
)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Method 3: SAE feature-level multilingual preservation. Learn a "
            "sparse dictionary over anchor hidden states, identify features "
            "important for the old language relative to the new language, and "
            "preserve only those feature activations during new-language training."
        )
    )
    ap.add_argument("--anchor_checkpoint", required=True)
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--eval_languages", nargs="+", default=["en", "zh", "fr"])
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--lambda_preserve", type=float, default=1.0)
    ap.add_argument("--dict_size", type=int, default=2048)
    ap.add_argument("--top_k_features", type=int, default=256)
    ap.add_argument(
        "--feature_score_mode",
        choices=["ratio", "old_fraction", "old_only"],
        default="old_fraction",
    )
    ap.add_argument("--sae_sample_batches", type=int, default=16)
    ap.add_argument("--sae_sample_batch", type=int, default=2)
    ap.add_argument("--sae_tokens_per_batch", type=int, default=128)
    ap.add_argument("--sae_epochs", type=int, default=10)
    ap.add_argument("--sae_batch", type=int, default=256)
    ap.add_argument("--sae_lr", type=float, default=1e-3)
    ap.add_argument("--sae_l1", type=float, default=1e-3)
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
        default="invariance_runs/advanced_methods/sae_feature_preservation",
    )
    ap.add_argument("--no_bf16", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.top_k_features <= 0:
        raise ValueError("--top_k_features must be > 0")

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

    print("Loading anchor to collect SAE activations...")
    anchor = load_model(
        args.anchor_checkpoint, device, use_bf16, trainable=False
    )
    old_acts = sample_hidden_tokens(
        anchor,
        old_train,
        layer=args.layer,
        batch_size=args.sae_sample_batch,
        max_batches=args.sae_sample_batches,
        tokens_per_batch=args.sae_tokens_per_batch,
        device=device,
        use_bf16=use_bf16,
        seed=args.seed + 11,
    )
    new_acts = sample_hidden_tokens(
        anchor,
        new_train_raw,
        layer=args.layer,
        batch_size=args.sae_sample_batch,
        max_batches=args.sae_sample_batches,
        tokens_per_batch=args.sae_tokens_per_batch,
        device=device,
        use_bf16=use_bf16,
        seed=args.seed + 22,
    )

    anchor_losses = {
        lang: evaluate(anchor, val[lang], args.eval_batch, device, use_bf16)
        for lang in args.eval_languages
    }
    del anchor
    torch.cuda.empty_cache()

    sae_acts = torch.cat([old_acts, new_acts], dim=0).float()
    center = sae_acts.mean(dim=0)
    sae_train = sae_acts - center.view(1, -1)

    print(
        f"Training SAE: samples={sae_train.shape[0]} "
        f"d_model={sae_train.shape[1]} dict={args.dict_size}"
    )
    sae = train_sparse_autoencoder(
        sae_train,
        dict_size=args.dict_size,
        epochs=args.sae_epochs,
        batch_size=args.sae_batch,
        lr=args.sae_lr,
        l1_lambda=args.sae_l1,
        device=device,
        seed=args.seed,
    )
    for p in sae.parameters():
        p.requires_grad_(False)
    sae.eval()

    print("Estimating old-language SAE feature importance...")
    old_imp = estimate_sae_feature_importance(
        args.anchor_checkpoint,
        sae,
        old_train,
        layer=args.layer,
        batch_size=args.importance_batch,
        max_batches=args.importance_batches,
        device=device,
        use_bf16=use_bf16,
        center=center,
    )
    print("Estimating new-language SAE feature importance...")
    new_imp = estimate_sae_feature_importance(
        args.anchor_checkpoint,
        sae,
        new_train_raw,
        layer=args.layer,
        batch_size=args.importance_batch,
        max_batches=args.importance_batches,
        device=device,
        use_bf16=use_bf16,
        center=center,
    )
    mask, feature_score = select_top_feature_mask(
        old_imp,
        new_imp,
        top_k=args.top_k_features,
        mode=args.feature_score_mode,
    )
    mask = mask.to(device)
    selected = int(mask.sum().item())

    torch.save(
        {
            "layer": args.layer,
            "dict_size": args.dict_size,
            "center": center,
            "sae_state_dict": sae.cpu().state_dict(),
            "old_feature_importance": old_imp,
            "new_feature_importance": new_imp,
            "feature_score": feature_score,
            "selected_mask": mask.cpu(),
            "feature_score_mode": args.feature_score_mode,
        },
        out_dir / "sae_preservation_artifact.pt",
    )
    sae = sae.to(device)

    anchor = load_model(
        args.anchor_checkpoint, device, use_bf16, trainable=False
    )
    model = load_model(
        args.anchor_checkpoint, device, use_bf16, trainable=True
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    opt.zero_grad(set_to_none=True)

    loader = make_loader(new_train, args.micro_batch)
    total_lm = 0.0
    total_pres = 0.0
    n_batches = 0
    tokens_seen = 0
    center_gpu = center.to(device).view(1, 1, -1)

    for i, (x,) in enumerate(loader):
        x = x.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda" and use_bf16),
            ):
                anchor_out = anchor(
                    input_ids=x,
                    output_hidden_states=True,
                    use_cache=False,
                )
            h_anchor = anchor_out.hidden_states[args.layer].float()
            z_anchor = sae.encode(h_anchor - center_gpu).detach()

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

        h_cur = out.hidden_states[args.layer].float()
        z_cur = sae.encode(h_cur - center_gpu)
        feature_delta = (z_cur - z_anchor).pow(2)
        preserve_loss = (
            feature_delta * mask.view(1, 1, -1)
        ).sum(dim=-1).mean() / max(selected, 1)

        loss = (
            out.loss + args.lambda_preserve * preserve_loss
        ) / args.grad_accum
        loss.backward()

        total_lm += float(out.loss.detach())
        total_pres += float(preserve_loss.detach())
        n_batches += 1
        tokens_seen += x.numel()

        do_step = ((i + 1) % args.grad_accum == 0) or (i + 1 == len(loader))
        if do_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if (i + 1) % 50 == 0:
            print(
                f"batch={i+1}/{len(loader)} "
                f"lm={total_lm/n_batches:.4f} "
                f"sae_pres={total_pres/n_batches:.6f}"
            )

    final_losses = {
        lang: evaluate(model, val[lang], args.eval_batch, device, use_bf16)
        for lang in args.eval_languages
    }

    row = {
        "method": "sae_feature_preservation",
        "seed": args.seed,
        "layer": args.layer,
        "lambda_preserve": args.lambda_preserve,
        "dict_size": args.dict_size,
        "top_k_features": selected,
        "feature_score_mode": args.feature_score_mode,
        "old_language": args.old_language,
        "new_language": args.new_language,
        "tokens_seen": tokens_seen,
        "mean_train_lm_loss": total_lm / max(n_batches, 1),
        "mean_preserve_loss": total_pres / max(n_batches, 1),
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

    print("\n=== SAE Feature Preservation ===")
    print(
        f"selected={selected}/{args.dict_size} "
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
