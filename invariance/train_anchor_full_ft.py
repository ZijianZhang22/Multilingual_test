import argparse
import csv
import json
from pathlib import Path

import torch

from advanced_preservation_utils import (
    evaluate,
    load_blocks,
    load_model,
    make_loader,
    set_seed,
    shuffled_blocks,
)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Clean Full-FT baseline from an existing stage-1 language anchor. "
            "Uses the exact same shuffled/truncated new-language blocks as the "
            "small preservation sweeps."
        )
    )
    ap.add_argument("--anchor_checkpoint", required=True)
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", required=True)
    ap.add_argument("--new_language", required=True)
    ap.add_argument("--eval_languages", nargs="+", default=None)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_shuffle_seed", type=int, default=1000)
    ap.add_argument("--max_train_blocks", type=int, default=400)
    ap.add_argument(
        "--out_dir",
        default="invariance_runs/small_importance_sweep/full_ft",
    )
    ap.add_argument("--no_bf16", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    eval_languages = args.eval_languages or [args.old_language, args.new_language]
    device = torch.device("cuda")
    use_bf16 = (not args.no_bf16) and torch.cuda.is_bf16_supported()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    new_train_raw = load_blocks(Path(args.data_dir) / f"{args.new_language}_train.pt")
    new_train = shuffled_blocks(
        new_train_raw, args.data_shuffle_seed, args.max_train_blocks
    )
    val = {
        lang: load_blocks(Path(args.data_dir) / f"{lang}_val.pt")
        for lang in eval_languages
    }

    model = load_model(args.anchor_checkpoint, device, use_bf16, trainable=True)

    anchor_losses = {
        lang: evaluate(model, val[lang], args.eval_batch, device, use_bf16)
        for lang in eval_languages
    }
    print("Anchor losses:", anchor_losses)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    opt.zero_grad(set_to_none=True)

    loader = make_loader(new_train, args.micro_batch)
    lm_total = 0.0
    n_batches = 0
    tokens_seen = 0

    for i, (x,) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and use_bf16),
        ):
            out = model(input_ids=x, labels=x, use_cache=False)
            loss = out.loss / args.grad_accum
        loss.backward()

        lm_total += float(out.loss.detach())
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
                f"lm={lm_total/max(n_batches,1):.4f}"
            )

    final_losses = {
        lang: evaluate(model, val[lang], args.eval_batch, device, use_bf16)
        for lang in eval_languages
    }

    row = {
        "method": "full_ft",
        "seed": args.seed,
        "old_language": args.old_language,
        "new_language": args.new_language,
        "max_train_blocks": args.max_train_blocks,
        "tokens_seen": tokens_seen,
        "mean_train_lm_loss": lm_total / max(n_batches, 1),
        "forgetting_loss_delta": (
            final_losses[args.old_language] - anchor_losses[args.old_language]
        ),
        "new_language_gain": (
            anchor_losses[args.new_language] - final_losses[args.new_language]
        ),
    }
    for lang in eval_languages:
        row[f"{lang}_anchor_loss"] = anchor_losses[lang]
        row[f"{lang}_final_loss"] = final_losses[lang]
        row[f"{lang}_loss_delta"] = final_losses[lang] - anchor_losses[lang]

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    manifest = vars(args).copy()
    manifest["eval_languages_resolved"] = eval_languages
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n=== Small Full FT baseline ===")
    print(
        f"forget={row['forgetting_loss_delta']:+.6f} "
        f"new_gain={row['new_language_gain']:+.6f}"
    )
    print(f"Saved: {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
