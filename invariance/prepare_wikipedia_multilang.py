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


def collect_train_val(lang, train_tokens, val_tokens, block_size, tokenizer, seed):
    """Collect validation first, then discard the boundary buffer and collect train from later articles."""
    n_val = val_tokens // block_size
    n_train = train_tokens // block_size
    if min(n_val, n_train) < 1:
        raise ValueError("train_tokens and val_tokens must each be >= block_size")

    val_blocks, train_blocks, buffer = [], [], []
    phase = "val"
    pbar = tqdm(total=n_val + n_train, desc=f"Collecting {lang}")
    prev_total = 0

    for ex in wikipedia_stream(lang, seed):
        text = ex.get("text", "")
        if not text:
            continue

        add_document_tokens(buffer, text, tokenizer)

        if phase == "val":
            pop_blocks(buffer, val_blocks, n_val, block_size)
            if len(val_blocks) >= n_val:
                buffer = []
                phase = "train"
        else:
            pop_blocks(buffer, train_blocks, n_train, block_size)

        current_total = len(val_blocks) + len(train_blocks)
        if current_total > prev_total:
            pbar.update(current_total - prev_total)
            prev_total = current_total

        if len(val_blocks) >= n_val and len(train_blocks) >= n_train:
            break

    pbar.close()
    if len(val_blocks) < n_val or len(train_blocks) < n_train:
        raise RuntimeError(
            f"Not enough {lang} text: val={len(val_blocks)}/{n_val}, "
            f"train={len(train_blocks)}/{n_train}"
        )

    return (
        torch.tensor(train_blocks, dtype=torch.int32),
        torch.tensor(val_blocks, dtype=torch.int32),
    )


def save_tensor(path: Path, tensor: torch.Tensor, block_size: int):
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
    ap.add_argument("--languages", nargs="+", default=["en", "zh", "fr"])
    ap.add_argument("--train_tokens", type=int, default=1_000_000,
                    help="Training-token budget PER language")
    ap.add_argument("--val_tokens", type=int, default=100_000,
                    help="Held-out validation-token budget PER language")
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", default="invariance_data/wiki")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    metadata = {
        "model_name": args.model_name,
        "languages": args.languages,
        "block_size": args.block_size,
        "train_tokens_requested_per_language": args.train_tokens,
        "val_tokens_requested_per_language": args.val_tokens,
        "files": {},
    }

    for i, lang in enumerate(args.languages):
        train, val = collect_train_val(
            lang,
            args.train_tokens,
            args.val_tokens,
            args.block_size,
            tok,
            args.seed + i * 17,
        )
        train_path = out / f"{lang}_train.pt"
        val_path = out / f"{lang}_val.pt"
        save_tensor(train_path, train, args.block_size)
        save_tensor(val_path, val, args.block_size)
        metadata["files"][lang] = {
            "train": str(train_path),
            "val": str(val_path),
            "train_tokens_actual": int(train.numel()),
            "val_tokens_actual": int(val.numel()),
        }

    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
