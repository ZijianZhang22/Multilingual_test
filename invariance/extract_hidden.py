import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class JsonlDataset(Dataset):
    def __init__(self, path):
        self.rows = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate(rows):
    return rows


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_file", default="invariance_data/xnli_probe.jsonl")
    ap.add_argument("--out_file", required=True)
    ap.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[0, 4, 8, 12, 16, 20, 24],
        help="Hidden-state indices; 0 is embeddings, later indices are transformer outputs",
    )
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--pool", choices=["mean", "last"], default="mean")
    ap.add_argument("--no_bf16", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    use_bf16 = (not args.no_bf16) and torch.cuda.is_bf16_supported()

    tok = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=(torch.bfloat16 if use_bf16 else torch.float32),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    ds = JsonlDataset(args.data_file)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    features = {layer: [] for layer in args.layers}
    labels, languages, splits, ids = [], [], [], []

    with torch.no_grad():
        for rows in loader:
            texts = [r["premise"] + "\n\n" + r["hypothesis"] for r in rows]
            enc = tok(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            n_states = len(hs)

            for layer in args.layers:
                idx = layer if layer >= 0 else n_states + layer
                if idx < 0 or idx >= n_states:
                    raise ValueError(
                        f"Requested layer {layer}, but model returned {n_states} hidden states"
                    )
                h = hs[idx]
                if args.pool == "mean":
                    pooled = mean_pool(h, enc["attention_mask"])
                else:
                    last_idx = enc["attention_mask"].sum(dim=1) - 1
                    pooled = h[
                        torch.arange(h.shape[0], device=device),
                        last_idx,
                    ]
                features[layer].append(pooled.float().cpu())

            labels.extend(int(r["label"]) for r in rows)
            languages.extend(r["language"] for r in rows)
            splits.extend(r["split"] for r in rows)
            ids.extend(r["example_id"] for r in rows)

    payload = {
        "checkpoint": args.checkpoint,
        "pool": args.pool,
        "layers": args.layers,
        "features": {
            str(k): torch.cat(v, dim=0) for k, v in features.items()
        },
        "labels": torch.tensor(labels, dtype=torch.long),
        "languages": languages,
        "splits": splits,
        "example_ids": ids,
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"Saved hidden-state features to {out_path}")
    for layer in args.layers:
        print(
            f"layer {layer}: "
            f"{tuple(payload['features'][str(layer)].shape)}"
        )


if __name__ == "__main__":
    main()
