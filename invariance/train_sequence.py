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


def train_one_language(
    model,
    blocks,
    *,
    lr,
    weight_decay,
    micro_batch,
    grad_accum,
    device,
    use_bf16,
):
    """Use constant LR and reset optimizer state for every language stage.

    This removes the confound where one across-stage cosine schedule gives the
    first language systematically larger learning rates.
    """
    loader = make_loader(blocks, micro_batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt.zero_grad(set_to_none=True)

    tokens_seen = 0
    for i, (x,) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and use_bf16),
        ):
            out = model(input_ids=x, labels=x, use_cache=False)
            loss = out.loss / grad_accum
        loss.backward()
        tokens_seen += x.numel()

        do_step = ((i + 1) % grad_accum == 0) or (i + 1 == len(loader))
        if do_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

    del opt
    return tokens_seen


def save_checkpoint(model, tokenizer, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument(
        "--sequences",
        nargs="+",
        default=["en,zh", "zh,en"],
        help='Comma-separated language orders, e.g. "en,zh" "zh,en".',
    )
    ap.add_argument("--out_dir", default="invariance_runs/sequence_seed0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument(
        "--eval_languages",
        nargs="+",
        default=None,
        help="Languages to evaluate after every stage. Defaults to languages in the training sequences.",
    )
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    use_bf16 = (not args.no_bf16) and torch.cuda.is_bf16_supported()
    data_dir = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sequences = [tuple(x.split(",")) for x in args.sequences]
    train_languages = sorted({lang for seq in sequences for lang in seq})
    eval_languages = args.eval_languages or train_languages
    train = {
        lang: load_blocks(data_dir / f"{lang}_train.pt")
        for lang in train_languages
    }
    val = {
        lang: load_blocks(data_dir / f"{lang}_val.pt")
        for lang in eval_languages
    }

    shuffled = {}
    for idx, lang in enumerate(train_languages):
        g = torch.Generator().manual_seed(args.seed + 1000 + idx)
        shuffled[lang] = train[lang][torch.randperm(len(train[lang]), generator=g)]

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = []
    manifest = {
        "model_name": args.model_name,
        "seed": args.seed,
        "lr": args.lr,
        "schedule": "constant_lr_optimizer_reset_each_language",
        "sequences": [list(s) for s in sequences],
        "eval_languages": list(eval_languages),
    }

    for sequence in sequences:
        set_seed(args.seed)
        name = "__".join(sequence)
        branch_dir = out_root / name
        branch_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=(torch.bfloat16 if use_bf16 else torch.float32),
            low_cpu_mem_usage=True,
        ).to(device)
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        save_checkpoint(model, tok, branch_dir / "stage0_base")

        # Stage-0 evaluation is required to distinguish true forgetting from
        # simple differences in the pretrained starting point.
        for eval_lang in eval_languages:
            loss = evaluate(model, val[eval_lang], args.eval_batch, device, use_bf16)
            rows.append({
                "sequence": name,
                "stage": 0,
                "trained_language": "base",
                "eval_language": eval_lang,
                "tokens_seen_in_stage": 0,
                "val_loss": loss,
            })
            print(
                f"[{name}] stage=0 trained=base "
                f"eval={eval_lang} loss={loss:.4f}"
            )

        for stage_idx, lang in enumerate(sequence, start=1):
            tokens_seen = train_one_language(
                model,
                shuffled[lang],
                lr=args.lr,
                weight_decay=args.weight_decay,
                micro_batch=args.micro_batch,
                grad_accum=args.grad_accum,
                device=device,
                use_bf16=use_bf16,
            )

            ckpt = branch_dir / f"stage{stage_idx}_{lang}"
            save_checkpoint(model, tok, ckpt)

            for eval_lang in eval_languages:
                loss = evaluate(model, val[eval_lang], args.eval_batch, device, use_bf16)
                rows.append({
                    "sequence": name,
                    "stage": stage_idx,
                    "trained_language": lang,
                    "eval_language": eval_lang,
                    "tokens_seen_in_stage": tokens_seen,
                    "val_loss": loss,
                })
                print(
                    f"[{name}] stage={stage_idx} trained={lang} "
                    f"eval={eval_lang} loss={loss:.4f}"
                )

            branch_rows = [r for r in rows if r["sequence"] == name]
            with (branch_dir / "sequence_metrics.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                fields = [
                    "sequence", "stage", "trained_language", "eval_language",
                    "tokens_seen_in_stage", "val_loss",
                ]
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(branch_rows)

        del model
        torch.cuda.empty_cache()

    with (out_root / "all_sequence_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "sequence", "stage", "trained_language", "eval_language",
            "tokens_seen_in_stage", "val_loss",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Done: {out_root}")


if __name__ == "__main__":
    main()
