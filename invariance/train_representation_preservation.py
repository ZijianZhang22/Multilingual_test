import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_blocks(path):
    return torch.load(path, map_location="cpu")["input_ids"].long()


def make_loader(blocks, batch_size):
    return DataLoader(
        TensorDataset(blocks),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate(model, blocks, batch_size, device, use_bf16):
    model.eval()
    total_loss = 0.0
    total_targets = 0
    for (x,) in make_loader(blocks, batch_size):
        x = x.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and use_bf16),
        ):
            out = model(input_ids=x, labels=x, use_cache=False)
        n = x.shape[0] * (x.shape[1] - 1)
        total_loss += float(out.loss) * n
        total_targets += n
    model.train()
    return total_loss / total_targets


def save_checkpoint(model, tokenizer, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def make_shared_random_basis(q_lang, rank, seed):
    """Sample an orthonormal rank-k basis inside the INLP residual complement."""
    d = q_lang.shape[0]
    rank = min(rank, d - q_lang.shape[1])
    if rank <= 0:
        raise ValueError("No room for a residual random subspace")

    g = torch.Generator(device=q_lang.device)
    g.manual_seed(seed)
    r = torch.randn(
        d, rank, device=q_lang.device, dtype=torch.float32, generator=g
    )
    r = r - q_lang @ (q_lang.T @ r)
    q, _ = torch.linalg.qr(r, mode="reduced")
    return q[:, :rank]


def projected_mse(
    delta,
    mode,
    q_lang,
    q_shared_random=None,
    q_transfer=None,
):
    """Per-dimension MSE so losses are comparable across subspace ranks."""
    delta = delta.float()

    if mode == "full":
        return delta.pow(2).mean()

    if mode == "lang":
        return (delta @ q_lang).pow(2).mean()

    if mode == "shared":
        lang_part = (delta @ q_lang) @ q_lang.T
        shared = delta - lang_part
        return shared.pow(2).sum(dim=-1).mean() / shared.shape[-1]

    if mode == "shared64":
        if q_shared_random is None:
            raise ValueError("shared64 requires q_shared_random")
        return (delta @ q_shared_random).pow(2).mean()

    if mode == "transfer64":
        if q_transfer is None:
            raise ValueError(
                "transfer64 requires --transferable_subspace_file"
            )
        return (delta @ q_transfer).pow(2).mean()

    raise ValueError(mode)


def make_curve_targets(n_batches, fractions):
    targets = []
    for frac in sorted(set(fractions)):
        if frac <= 0 or frac > 1:
            raise ValueError("--curve_fractions values must be in (0, 1]")
        batch_target = max(1, int(np.ceil(frac * n_batches)))
        targets.append((float(frac), batch_target))
    return targets


def train_intervention(
    model,
    anchor,
    blocks,
    *,
    mode,
    preserve_lambda,
    layer,
    q_lang,
    q_shared_random,
    q_transfer,
    lr,
    weight_decay,
    micro_batch,
    grad_accum,
    device,
    use_bf16,
    curve_fractions=None,
    curve_callback=None,
):
    loader = make_loader(blocks, micro_batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt.zero_grad(set_to_none=True)

    total_lm = 0.0
    total_pres = 0.0
    n_batches = 0
    tokens_seen = 0
    curve_targets = make_curve_targets(len(loader), curve_fractions or [])
    next_curve = 0

    anchor.eval()
    model.train()

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
            h_anchor = anchor_out.hidden_states[layer].detach().float()

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
            lm_loss = out.loss

        delta = out.hidden_states[layer].float() - h_anchor
        pres_loss = projected_mse(
            delta,
            mode,
            q_lang,
            q_shared_random=q_shared_random,
            q_transfer=q_transfer,
        )

        ((lm_loss + preserve_lambda * pres_loss) / grad_accum).backward()

        total_lm += float(lm_loss.detach())
        total_pres += float(pres_loss.detach())
        n_batches += 1
        tokens_seen += x.numel()

        do_step = ((i + 1) % grad_accum == 0) or (i + 1 == len(loader))
        if do_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            while (
                next_curve < len(curve_targets)
                and (i + 1) >= curve_targets[next_curve][1]
            ):
                frac = curve_targets[next_curve][0]
                if curve_callback is not None:
                    curve_callback(model, frac, tokens_seen)
                model.train()
                next_curve += 1

    del opt
    return {
        "tokens_seen": tokens_seen,
        "mean_train_lm_loss": total_lm / max(n_batches, 1),
        "mean_preserve_loss": total_pres / max(n_batches, 1),
    }


def train_full_ft(
    model,
    blocks,
    *,
    lr,
    weight_decay,
    micro_batch,
    grad_accum,
    device,
    use_bf16,
    curve_fractions=None,
    curve_callback=None,
):
    loader = make_loader(blocks, micro_batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt.zero_grad(set_to_none=True)

    total_lm = 0.0
    n_batches = 0
    tokens_seen = 0
    curve_targets = make_curve_targets(len(loader), curve_fractions or [])
    next_curve = 0

    for i, (x,) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and use_bf16),
        ):
            out = model(input_ids=x, labels=x, use_cache=False)
            lm_loss = out.loss

        (lm_loss / grad_accum).backward()
        total_lm += float(lm_loss.detach())
        n_batches += 1
        tokens_seen += x.numel()

        do_step = ((i + 1) % grad_accum == 0) or (i + 1 == len(loader))
        if do_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            while (
                next_curve < len(curve_targets)
                and (i + 1) >= curve_targets[next_curve][1]
            ):
                frac = curve_targets[next_curve][0]
                if curve_callback is not None:
                    curve_callback(model, frac, tokens_seen)
                model.train()
                next_curve += 1

    del opt
    return {
        "tokens_seen": tokens_seen,
        "mean_train_lm_loss": total_lm / max(n_batches, 1),
        "mean_preserve_loss": 0.0,
    }


def load_model(checkpoint, device, use_bf16):
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    return model


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Causal intervention: preserve selected hidden-state components while "
            "learning a second language, then measure retention/plasticity/transfer."
        )
    )
    ap.add_argument("--anchor_checkpoint", required=True)
    ap.add_argument(
        "--subspace_file",
        default="invariance_analysis/subspace_seed0/reference_inlp_language_subspace.pt",
    )
    ap.add_argument(
        "--transferable_subspace_file",
        default=None,
        help="Optional .pt file containing transferable_subspace_basis.",
    )
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--eval_languages", nargs="+", default=["en", "zh", "fr"])
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["full_ft", "full", "shared", "lang", "shared64"],
        choices=[
            "full_ft", "full", "shared", "lang", "shared64", "transfer64"
        ],
    )
    ap.add_argument("--lambdas", nargs="+", type=float, default=[1.0])
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--shared64_rank", type=int, default=64)
    ap.add_argument("--out_dir", default="invariance_runs/interventions_en_to_zh_seed0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--data_shuffle_seed",
        type=int,
        default=None,
        help=(
            "Shuffle seed for the new-language blocks. For EN->ZH seed 0 use "
            "1001; for ZH->EN seed 0 use 1000 to match train_sequence.py."
        ),
    )
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument(
        "--curve_fractions",
        type=float,
        nargs="*",
        default=[],
        help="Optional learning-curve fractions, e.g. 0.1 0.25 0.5 0.75 1.0.",
    )
    ap.add_argument(
        "--max_train_blocks",
        type=int,
        default=None,
        help="Optional smoke-test limit after shuffling; omit for the full stage.",
    )
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--save_checkpoints", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    use_bf16 = (not args.no_bf16) and torch.cuda.is_bf16_supported()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = torch.load(args.subspace_file, map_location="cpu")
    q_lang = sub["language_subspace_basis"].float().to(device)
    if int(sub["layer"]) != args.layer:
        raise ValueError(
            f"Language subspace is layer {sub['layer']}, requested {args.layer}"
        )

    q_shared_random = make_shared_random_basis(
        q_lang, args.shared64_rank, args.seed + 777
    )

    q_transfer = None
    if "transfer64" in args.methods:
        if args.transferable_subspace_file is None:
            raise ValueError(
                "transfer64 requested but --transferable_subspace_file is missing"
            )
        transfer = torch.load(args.transferable_subspace_file, map_location="cpu")
        if int(transfer["layer"]) != args.layer:
            raise ValueError("Transferable subspace layer does not match")
        q_transfer = transfer["transferable_subspace_basis"].float().to(device)

    train_blocks = load_blocks(
        Path(args.data_dir) / f"{args.new_language}_train.pt"
    )
    shuffle_seed = (
        args.data_shuffle_seed
        if args.data_shuffle_seed is not None
        else args.seed + 1001
    )
    g = torch.Generator().manual_seed(shuffle_seed)
    train_blocks = train_blocks[
        torch.randperm(len(train_blocks), generator=g)
    ]
    if args.max_train_blocks is not None:
        train_blocks = train_blocks[: args.max_train_blocks]

    val = {
        lang: load_blocks(Path(args.data_dir) / f"{lang}_val.pt")
        for lang in args.eval_languages
    }

    tok = AutoTokenizer.from_pretrained(args.anchor_checkpoint, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("Loading frozen anchor model...")
    anchor = load_model(args.anchor_checkpoint, device, use_bf16)
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)

    anchor_losses = {}
    for lang in args.eval_languages:
        anchor_losses[lang] = evaluate(
            anchor, val[lang], args.eval_batch, device, use_bf16
        )
        print(f"[anchor] eval={lang} loss={anchor_losses[lang]:.4f}")

    rows = []
    curve_rows = []

    configs = []
    for method in args.methods:
        if method == "full_ft":
            configs.append((method, 0.0))
        else:
            for lam in args.lambdas:
                configs.append((method, lam))

    for method, lam in configs:
        set_seed(args.seed)
        print(f"\n=== method={method} lambda={lam:g} ===")
        model = load_model(args.anchor_checkpoint, device, use_bf16)

        if args.curve_fractions:
            curve_rows.append({
                "method": method,
                "lambda": lam,
                "seed": args.seed,
                "fraction": 0.0,
                "tokens_seen": 0,
                **{
                    f"{lang}_loss": anchor_losses[lang]
                    for lang in args.eval_languages
                },
                "forgetting_loss_delta": 0.0,
                "new_language_gain": 0.0,
            })

        def curve_callback(cur_model, fraction, tokens_seen):
            losses = {
                lang: evaluate(
                    cur_model, val[lang], args.eval_batch, device, use_bf16
                )
                for lang in args.eval_languages
            }
            curve_rows.append({
                "method": method,
                "lambda": lam,
                "seed": args.seed,
                "fraction": fraction,
                "tokens_seen": tokens_seen,
                **{f"{lang}_loss": losses[lang] for lang in args.eval_languages},
                "forgetting_loss_delta": (
                    losses[args.old_language] - anchor_losses[args.old_language]
                ),
                "new_language_gain": (
                    anchor_losses[args.new_language] - losses[args.new_language]
                ),
            })
            print(
                f"[curve {method} lam={lam:g} frac={fraction:.2f}] "
                f"forget={losses[args.old_language]-anchor_losses[args.old_language]:+.5f} "
                f"new_gain={anchor_losses[args.new_language]-losses[args.new_language]:+.5f}"
            )

        if method == "full_ft":
            train_stats = train_full_ft(
                model,
                train_blocks,
                lr=args.lr,
                weight_decay=args.weight_decay,
                micro_batch=args.micro_batch,
                grad_accum=args.grad_accum,
                device=device,
                use_bf16=use_bf16,
                curve_fractions=args.curve_fractions,
                curve_callback=curve_callback,
            )
        else:
            train_stats = train_intervention(
                model,
                anchor,
                train_blocks,
                mode=method,
                preserve_lambda=lam,
                layer=args.layer,
                q_lang=q_lang,
                q_shared_random=q_shared_random,
                q_transfer=q_transfer,
                lr=args.lr,
                weight_decay=args.weight_decay,
                micro_batch=args.micro_batch,
                grad_accum=args.grad_accum,
                device=device,
                use_bf16=use_bf16,
                curve_fractions=args.curve_fractions,
                curve_callback=curve_callback,
            )

        eval_losses = {}
        for lang in args.eval_languages:
            loss = evaluate(model, val[lang], args.eval_batch, device, use_bf16)
            eval_losses[lang] = loss
            print(
                f"[{method} lambda={lam:g}] eval={lang} "
                f"loss={loss:.4f} delta_from_anchor={loss-anchor_losses[lang]:+.4f}"
            )

        row = {
            "method": method,
            "lambda": lam,
            "seed": args.seed,
            "layer": args.layer,
            "old_language": args.old_language,
            "new_language": args.new_language,
            "tokens_seen": train_stats["tokens_seen"],
            "mean_train_lm_loss": train_stats["mean_train_lm_loss"],
            "mean_preserve_loss": train_stats["mean_preserve_loss"],
            "old_anchor_loss": anchor_losses[args.old_language],
            "old_final_loss": eval_losses[args.old_language],
            "forgetting_loss_delta": (
                eval_losses[args.old_language] - anchor_losses[args.old_language]
            ),
            "new_anchor_loss": anchor_losses[args.new_language],
            "new_final_loss": eval_losses[args.new_language],
            "new_language_gain": (
                anchor_losses[args.new_language] - eval_losses[args.new_language]
            ),
        }
        for lang in args.eval_languages:
            row[f"{lang}_anchor_loss"] = anchor_losses[lang]
            row[f"{lang}_final_loss"] = eval_losses[lang]
            row[f"{lang}_loss_delta"] = (
                eval_losses[lang] - anchor_losses[lang]
            )
        rows.append(row)

        if args.save_checkpoints:
            name = method if method == "full_ft" else f"{method}_lam{lam:g}"
            save_checkpoint(model, tok, out_dir / "checkpoints" / name)

        del model
        torch.cuda.empty_cache()

    fields = sorted({k for r in rows for k in r.keys()})
    with (out_dir / "intervention_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if curve_rows:
        curve_fields = sorted({k for r in curve_rows for k in r.keys()})
        with (out_dir / "learning_curves.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=curve_fields)
            w.writeheader()
            w.writerows(curve_rows)

    ranked = sorted(
        rows,
        key=lambda r: (r["forgetting_loss_delta"], -r["new_language_gain"]),
    )
    print("\n=== Retention / plasticity summary ===")
    for r in ranked:
        print(
            f"{r['method']:<10} lam={r['lambda']:<6g} "
            f"forget={r['forgetting_loss_delta']:+.5f} "
            f"new_gain={r['new_language_gain']:+.5f} "
            f"train_lm={r['mean_train_lm_loss']:.4f} "
            f"pres={r['mean_preserve_loss']:.6f}"
        )

    manifest = {
        "anchor_checkpoint": args.anchor_checkpoint,
        "subspace_file": args.subspace_file,
        "transferable_subspace_file": args.transferable_subspace_file,
        "old_language": args.old_language,
        "new_language": args.new_language,
        "eval_languages": args.eval_languages,
        "methods": args.methods,
        "lambdas": args.lambdas,
        "curve_fractions": args.curve_fractions,
        "layer": args.layer,
        "shared64_rank": args.shared64_rank,
        "seed": args.seed,
        "data_shuffle_seed": shuffle_seed,
        "preservation_inputs": (
            "new-language training blocks only; no old-language replay examples"
        ),
        "loss_normalization": "per represented dimension",
        "max_train_blocks": args.max_train_blocks,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {out_dir / 'intervention_metrics.csv'}")
    if curve_rows:
        print(f"Saved: {out_dir / 'learning_curves.csv'}")


if __name__ == "__main__":
    main()
