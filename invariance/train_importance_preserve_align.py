import argparse
import csv
import json
from pathlib import Path

import torch

from advanced_preservation_utils import (
    batch_pairs,
    estimate_hidden_gradient_importance,
    evaluate,
    load_aligned_pairs,
    load_blocks,
    load_model,
    load_tokenizer,
    make_loader,
    make_stability_weights,
    pooled_hidden_from_texts,
    set_seed,
    shuffled_blocks,
)


def normalized_alignment_loss(current, target):
    current = torch.nn.functional.normalize(current.float(), dim=-1)
    target = torch.nn.functional.normalize(target.float(), dim=-1)
    return (current - target).pow(2).sum(dim=-1).mean()


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Method 1: importance-weighted representation preservation plus "
            "cross-lingual semantic alignment."
        )
    )
    ap.add_argument("--anchor_checkpoint", required=True)
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument(
        "--aligned_data_file",
        default="invariance_analysis/causal_representation_suite/xnli_aligned.jsonl",
    )
    ap.add_argument("--transferable_subspace_file", default=None)
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--eval_languages", nargs="+", default=["en", "zh", "fr"])
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--lambda_preserve", type=float, default=1.0)
    ap.add_argument("--lambda_align", type=float, default=1.0)
    ap.add_argument(
        "--importance_mode",
        choices=["ratio", "old_fraction", "old_only"],
        default="old_fraction",
    )
    ap.add_argument(
        "--weight_control",
        choices=["importance", "uniform", "shuffled"],
        default="importance",
        help=(
            "Which coordinate weights to use for representation preservation. "
            "'importance' uses the estimated stability weights; 'uniform' uses "
            "all-ones weights with the same mean scale; 'shuffled' randomly "
            "permutes the estimated weights, preserving their exact distribution "
            "while destroying coordinate identity."
        ),
    )
    ap.add_argument(
        "--weight_shuffle_seed",
        type=int,
        default=2026,
        help="Seed used only for the shuffled-weight control.",
    )
    ap.add_argument("--importance_batches", type=int, default=8)
    ap.add_argument("--importance_batch", type=int, default=2)
    ap.add_argument("--alignment_batch", type=int, default=8)
    ap.add_argument("--alignment_max_length", type=int, default=256)
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
        default="invariance_runs/advanced_methods/importance_align",
    )
    ap.add_argument("--no_bf16", action="store_true")
    args = ap.parse_args()

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

    print("Estimating old-language hidden importance...")
    old_imp = estimate_hidden_gradient_importance(
        args.anchor_checkpoint,
        old_train,
        layer=args.layer,
        batch_size=args.importance_batch,
        max_batches=args.importance_batches,
        device=device,
        use_bf16=use_bf16,
    )
    print("Estimating new-language hidden importance...")
    new_imp = estimate_hidden_gradient_importance(
        args.anchor_checkpoint,
        new_train_raw,
        layer=args.layer,
        batch_size=args.importance_batch,
        max_batches=args.importance_batches,
        device=device,
        use_bf16=use_bf16,
    )
    base_weights = make_stability_weights(
        old_imp, new_imp, mode=args.importance_mode
    )

    if args.weight_control == "importance":
        weights = base_weights.clone()
    elif args.weight_control == "uniform":
        # make_stability_weights normalizes mean weight to 1.0, so an all-ones
        # vector is the matched global-strength control.
        weights = torch.ones_like(base_weights)
    elif args.weight_control == "shuffled":
        # Preserve the exact empirical weight distribution while destroying
        # which hidden coordinate receives which weight.
        g = torch.Generator(device="cpu").manual_seed(args.weight_shuffle_seed)
        perm = torch.randperm(base_weights.numel(), generator=g)
        weights = base_weights[perm]
    else:
        raise ValueError(args.weight_control)

    weights = weights.to(device)

    torch.save(
        {
            "layer": args.layer,
            "old_importance": old_imp,
            "new_importance": new_imp,
            "base_importance_weights": base_weights.cpu(),
            "stability_weights": weights.cpu(),
            "importance_mode": args.importance_mode,
            "weight_control": args.weight_control,
            "weight_shuffle_seed": args.weight_shuffle_seed,
        },
        out_dir / "importance_weights.pt",
    )

    use_alignment = args.lambda_align != 0.0
    q_transfer = None
    pair_batches = None
    tok = None

    if use_alignment:
        if args.transferable_subspace_file:
            payload = torch.load(args.transferable_subspace_file, map_location="cpu")
            q_transfer = payload["transferable_subspace_basis"].float().to(device)
            if int(payload["layer"]) != args.layer:
                raise ValueError("Transferable subspace layer mismatch")

        pairs = load_aligned_pairs(
            args.aligned_data_file, args.old_language, args.new_language
        )
        pair_batches = batch_pairs(pairs, args.alignment_batch, args.seed + 4242)
        tok = load_tokenizer(args.anchor_checkpoint)
    else:
        print("lambda_align=0: skipping aligned-XNLI loading and alignment forward passes.")
    anchor = load_model(
        args.anchor_checkpoint, device, use_bf16, trainable=False
    )
    model = load_model(
        args.anchor_checkpoint, device, use_bf16, trainable=True
    )

    anchor_losses = {
        lang: evaluate(anchor, val[lang], args.eval_batch, device, use_bf16)
        for lang in args.eval_languages
    }
    print("Anchor losses:", anchor_losses)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    opt.zero_grad(set_to_none=True)

    lm_total = 0.0
    preserve_total = 0.0
    align_total = 0.0
    n_batches = 0
    tokens_seen = 0

    loader = make_loader(new_train, args.micro_batch)
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
            h_anchor = anchor_out.hidden_states[args.layer].detach().float()

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
        preserve_loss = (
            (h_cur - h_anchor).pow(2) * weights.view(1, 1, -1)
        ).mean()

        if use_alignment:
            pair_batch = pair_batches[i % len(pair_batches)]
            old_texts = [p[0] for p in pair_batch]
            new_texts = [p[1] for p in pair_batch]

            old_rep = pooled_hidden_from_texts(
                anchor,
                tok,
                old_texts,
                layer=args.layer,
                max_length=args.alignment_max_length,
                device=device,
                use_bf16=use_bf16,
                no_grad=True,
            ).detach()
            new_rep = pooled_hidden_from_texts(
                model,
                tok,
                new_texts,
                layer=args.layer,
                max_length=args.alignment_max_length,
                device=device,
                use_bf16=use_bf16,
                no_grad=False,
            )

            if q_transfer is not None:
                old_rep = old_rep.float() @ q_transfer
                new_rep = new_rep.float() @ q_transfer

            align_loss = normalized_alignment_loss(new_rep, old_rep)
        else:
            align_loss = out.loss.new_zeros(())

        loss = (
            out.loss
            + args.lambda_preserve * preserve_loss
            + args.lambda_align * align_loss
        ) / args.grad_accum
        loss.backward()

        lm_total += float(out.loss.detach())
        preserve_total += float(preserve_loss.detach())
        align_total += float(align_loss.detach())
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
                f"lm={lm_total/n_batches:.4f} "
                f"pres={preserve_total/n_batches:.6f} "
                f"align={align_total/n_batches:.6f}"
            )

    final_losses = {
        lang: evaluate(model, val[lang], args.eval_batch, device, use_bf16)
        for lang in args.eval_languages
    }

    row = {
        "method": "importance_preserve_align",
        "seed": args.seed,
        "layer": args.layer,
        "lambda_preserve": args.lambda_preserve,
        "lambda_align": args.lambda_align,
        "importance_mode": args.importance_mode,
        "weight_control": args.weight_control,
        "weight_shuffle_seed": args.weight_shuffle_seed,
        "old_language": args.old_language,
        "new_language": args.new_language,
        "tokens_seen": tokens_seen,
        "mean_train_lm_loss": lm_total / max(n_batches, 1),
        "mean_preserve_loss": preserve_total / max(n_batches, 1),
        "mean_align_loss": align_total / max(n_batches, 1),
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

    print(f"\n=== Preservation control: {args.weight_control} ===")
    print(
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
