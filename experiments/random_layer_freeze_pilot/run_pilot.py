#!/usr/bin/env python3
import argparse, csv, json, random, sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from invariance.train_sequence import load_blocks, make_loader, evaluate  # noqa: E402


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Cannot find transformer blocks (expected model.model.layers or model.transformer.h).")


def load_model(path, device, use_bf16):
    m = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    m.config.use_cache = False
    return m


class RandomLayerMask:
    """Freeze a fixed random fraction of parameter ELEMENTS inside one block."""
    def __init__(self, layer, ratio, seed):
        self.handles, self.entries = [], []
        self.total = self.frozen = 0
        self.ratio = float(ratio)
        if self.ratio == 1.0:
            for p in layer.parameters():
                self.total += p.numel(); self.frozen += p.numel(); p.requires_grad_(False)
            return
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for p in layer.parameters():
            self.total += p.numel()
            if self.ratio <= 0:
                continue
            train_cpu = torch.rand(p.shape, generator=gen) >= self.ratio
            self.frozen += int((~train_cpu).sum())
            train_mask = train_cpu.to(p.device)
            ref = p.detach().clone()
            self.handles.append(p.register_hook(lambda g, m=train_mask: g * m.to(g.dtype)))
            self.entries.append((p, train_mask, ref))

    @property
    def actual_ratio(self):
        return self.frozen / max(self.total, 1)

    @torch.no_grad()
    def restore(self):
        # AdamW decoupled weight decay can move zero-gradient entries, so restore exactly.
        for p, train_mask, ref in self.entries:
            frozen = ~train_mask
            p.data[frozen] = ref[frozen]

    def close(self):
        for h in self.handles:
            h.remove()


def train(model, blocks, args, device, use_bf16, mask=None):
    opt_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    opt.zero_grad(set_to_none=True)
    tokens = 0
    loader = make_loader(blocks, args.micro_batch)
    for i, (x,) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(input_ids=x, labels=x, use_cache=False).loss / args.grad_accum
        loss.backward(); tokens += x.numel()
        if (i + 1) % args.grad_accum == 0 or i + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            opt.step()
            if mask: mask.restore()
            opt.zero_grad(set_to_none=True)
    return tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anchor_checkpoint", default="invariance_runs/sequence_seed0/en__zh/stage1_en")
    p.add_argument("--data_dir", default="invariance_data/wiki")
    p.add_argument("--old_language", default="en"); p.add_argument("--new_language", default="zh")
    p.add_argument("--layers", type=int, nargs="+", default=[12, 20, 24])
    p.add_argument("--freeze_ratios", type=float, nargs="+", default=[0.2, 0.8, 1.0])
    p.add_argument("--train_fraction", type=float, default=0.20)
    p.add_argument("--eval_max_blocks", type=int, default=128)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--data_shuffle_seed", type=int, default=1001)
    p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--micro_batch", type=int, default=4); p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--eval_batch", type=int, default=8)
    p.add_argument("--out_dir", default="random_layer_freeze_runs/en_to_zh_seed0")
    p.add_argument("--skip_full_ft", action="store_true")
    p.add_argument("--save_checkpoints", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    if not 0 < args.train_fraction <= 1: raise ValueError("train_fraction must be in (0,1]")
    if any(r < 0 or r > 1 for r in args.freeze_ratios): raise ValueError("freeze ratios must be in [0,1]")

    device = torch.device("cuda")
    bf16 = torch.cuda.is_bf16_supported()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    new_train = load_blocks(Path(args.data_dir) / f"{args.new_language}_train.pt")
    old_val = load_blocks(Path(args.data_dir) / f"{args.old_language}_val.pt")
    new_val = load_blocks(Path(args.data_dir) / f"{args.new_language}_val.pt")
    if args.eval_max_blocks > 0:
        old_val, new_val = old_val[:args.eval_max_blocks], new_val[:args.eval_max_blocks]
    g = torch.Generator().manual_seed(args.data_shuffle_seed)
    perm = torch.randperm(len(new_train), generator=g)
    n = max(1, round(len(new_train) * args.train_fraction))
    new_train = new_train[perm[:n]]

    tok = AutoTokenizer.from_pretrained(args.anchor_checkpoint, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    anchor = load_model(args.anchor_checkpoint, device, bf16)
    n_layers = len(get_layers(anchor))
    for l in args.layers:
        if not 1 <= l <= n_layers: raise ValueError(f"layer {l} invalid; model has {n_layers} blocks")
    a_old = evaluate(anchor, old_val, args.eval_batch, device, bf16)
    a_new = evaluate(anchor, new_val, args.eval_batch, device, bf16)
    print(f"[anchor] old={a_old:.6f} new={a_new:.6f}; train_blocks={len(new_train)}")
    del anchor; torch.cuda.empty_cache()

    jobs = []
    if not args.skip_full_ft: jobs.append(("full_ft", None, 0.0))
    jobs += [(f"layer{l}_freeze{int(r*100)}", l, r) for l in args.layers for r in args.freeze_ratios]

    rows = []
    for j, (name, layer_no, ratio) in enumerate(jobs, 1):
        print(f"\n=== [{j}/{len(jobs)}] {name} ===")
        seed_all(args.seed)
        model = load_model(args.anchor_checkpoint, device, bf16)
        layers = get_layers(model)
        mask = None
        actual = 0.0
        layer_params = 0
        if layer_no is not None:
            mask_seed = args.seed + 100000 + layer_no * 1000 + int(ratio * 100)
            mask = RandomLayerMask(layers[layer_no - 1], ratio, mask_seed)
            actual, layer_params = mask.actual_ratio, mask.total
            print(f"layer={layer_no} requested={ratio:.2f} actual={actual:.4f} params={layer_params:,}")

        tokens = train(model, new_train, args, device, bf16, mask)
        p_old = evaluate(model, old_val, args.eval_batch, device, bf16)
        p_new = evaluate(model, new_val, args.eval_batch, device, bf16)
        forget, gain = p_old - a_old, a_new - p_new
        print(f"forget={forget:+.6f} new_gain={gain:+.6f}")

        if args.save_checkpoints:
            d = out / "checkpoints" / name; d.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(d); tok.save_pretrained(d)

        rows.append({
            "condition": name, "target_layer_1based": "" if layer_no is None else layer_no,
            "freeze_ratio_requested": ratio, "freeze_ratio_actual": actual,
            "layer_parameter_count": layer_params, "train_fraction": args.train_fraction,
            "train_blocks": len(new_train), "tokens_seen": tokens,
            "anchor_old_loss": a_old, "anchor_new_loss": a_new,
            "post_old_loss": p_old, "post_new_loss": p_new,
            "forgetting": forget, "new_language_gain": gain,
        })
        if mask: mask.close()
        del model; torch.cuda.empty_cache()
        with (out / "results_partial.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    base = next((r for r in rows if r["condition"] == "full_ft"), None)
    for r in rows:
        r["forgetting_reduction_vs_full_ft"] = "" if base is None else base["forgetting"] - r["forgetting"]
        r["plasticity_cost_vs_full_ft"] = "" if base is None else base["new_language_gain"] - r["new_language_gain"]

    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (out / "manifest.json").write_text(json.dumps(vars(args), indent=2))

    print("\n=== summary (sorted by forgetting) ===")
    for r in sorted(rows, key=lambda x: x["forgetting"]):
        print(f"{r['condition']:22s} forget={r['forgetting']:+.6f} gain={r['new_language_gain']:+.6f}")
    print(f"\nSaved: {out/'results.csv'}")


if __name__ == "__main__":
    main()
