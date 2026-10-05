import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def wikipedia_stream(lang: str, seed: int, buffer_size: int = 10_000):
    ds = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{lang}",
        split="train",
        streaming=True,
    )
    return ds.shuffle(seed=seed, buffer_size=buffer_size)


def add_document_tokens(buffer, text, tokenizer):
    ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    if not ids:
        return
    buffer.extend(ids)
    if tokenizer.eos_token_id is not None:
        buffer.append(tokenizer.eos_token_id)


def pop_blocks(buffer, blocks, n_blocks, block_size):
    while len(buffer) >= block_size and len(blocks) < n_blocks:
        blocks.append(buffer[:block_size])
        del buffer[:block_size]


def collect_blocks(lang, token_budget, block_size, tokenizer, seed):
    n_blocks = token_budget // block_size
    if n_blocks < 1:
        raise ValueError("token_budget must be at least block_size")

    blocks, buffer = [], []
    pbar = tqdm(total=n_blocks, desc=f"Collecting {lang}")
    prev = 0

    for ex in wikipedia_stream(lang, seed):
        text = ex.get("text", "")
        if not text:
            continue
        add_document_tokens(buffer, text, tokenizer)
        pop_blocks(buffer, blocks, n_blocks, block_size)
        if len(blocks) != prev:
            pbar.update(len(blocks) - prev)
            prev = len(blocks)
        if len(blocks) >= n_blocks:
            break
    pbar.close()

    if len(blocks) < n_blocks:
        raise RuntimeError(f"Not enough {lang} text: {len(blocks)}/{n_blocks} blocks")
    return torch.tensor(blocks, dtype=torch.int32)


def collect_target_train_val(lang, train_tokens, val_tokens, block_size, tokenizer, seed):
    """Validation first; training begins from later articles to avoid overlap."""
    n_val = val_tokens // block_size
    n_train = train_tokens // block_size
    if min(n_val, n_train) < 1:
        raise ValueError("train_tokens and val_tokens must each be >= block_size")

    val_blocks, train_blocks, buffer = [], [], []
    phase = "val"
    pbar_val = tqdm(total=n_val, desc=f"Collecting {lang} val")
    pbar_train = None
    prev_val = prev_train = 0

    for ex in wikipedia_stream(lang, seed):
        text = ex.get("text", "")
        if not text:
            continue

        if phase == "val":
            add_document_tokens(buffer, text, tokenizer)
            pop_blocks(buffer, val_blocks, n_val, block_size)
            if len(val_blocks) != prev_val:
                pbar_val.update(len(val_blocks) - prev_val)
                prev_val = len(val_blocks)
            if len(val_blocks) >= n_val:
                buffer = []
                phase = "train"
                pbar_val.close()
                pbar_train = tqdm(total=n_train, desc=f"Collecting {lang} train")
            continue

        add_document_tokens(buffer, text, tokenizer)
        pop_blocks(buffer, train_blocks, n_train, block_size)
        if len(train_blocks) != prev_train:
            pbar_train.update(len(train_blocks) - prev_train)
            prev_train = len(train_blocks)
        if len(train_blocks) >= n_train:
            break

    if pbar_train is not None:
        pbar_train.close()

    if len(val_blocks) < n_val or len(train_blocks) < n_train:
        raise RuntimeError(
            f"Not enough target text: val={len(val_blocks)}/{n_val}, train={len(train_blocks)}/{n_train}"
        )

    return (
        torch.tensor(train_blocks, dtype=torch.int32),
        torch.tensor(val_blocks, dtype=torch.int32),
    )


def save_tensor(path: Path, tensor: torch.Tensor, block_size: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": tensor,
            "num_blocks": int(tensor.shape[0]),
            "block_size": block_size,
            "num_tokens": int(tensor.numel()),
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out_dir", default="prepared")
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--source_tokens", type=int, default=5_000_000,
                    help="Token budget PER source language")
    ap.add_argument("--target_train_tokens", type=int, default=2_000_000)
    ap.add_argument("--target_val_tokens", type=int, default=250_000)
    ap.add_argument("--en_lang", default="en")
    ap.add_argument("--zh_lang", default="zh")
    ap.add_argument("--target_lang", default="sw")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    en = collect_blocks(args.en_lang, args.source_tokens, args.block_size, tok, args.seed + 1)
    zh = collect_blocks(args.zh_lang, args.source_tokens, args.block_size, tok, args.seed + 2)
    target_train, target_val = collect_target_train_val(
        args.target_lang,
        args.target_train_tokens,
        args.target_val_tokens,
        args.block_size,
        tok,
        args.seed + 3,
    )

    save_tensor(out / "en.pt", en, args.block_size)
    save_tensor(out / "zh.pt", zh, args.block_size)
    save_tensor(out / "target_train.pt", target_train, args.block_size)
    save_tensor(out / "target_val.pt", target_val, args.block_size)

    metadata = {
        "model_name": args.model_name,
        "block_size": args.block_size,
        "source_tokens_requested_per_language": args.source_tokens,
        "en_tokens_actual": int(en.numel()),
        "zh_tokens_actual": int(zh.numel()),
        "target_train_tokens_requested": args.target_train_tokens,
        "target_train_tokens_actual": int(target_train.numel()),
        "target_val_tokens_requested": args.target_val_tokens,
        "target_val_tokens_actual": int(target_val.numel()),
        "languages": {"A": args.en_lang, "B": args.zh_lang, "C": args.target_lang},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
