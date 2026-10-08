#!/usr/bin/env python3
"""Single-A100 BF16 full-parameter Qwen2.5-7B EN->ZH training using 8-bit AdamW.

No LoRA or quantized model weights. Quantized OPTIMIZER STATES are not an
exact numerical reproduction of the standard AdamW 0.5B reference.
"""
import argparse
import gc
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from invariance.train_sequence import evaluate, load_blocks, set_seed


def complete_checkpoint(path):
    """Reject incomplete sharded checkpoints left by interrupted saves."""
    p = Path(path)
    if not (p / "config.json").is_file():
        return False
    index = p / "model.safetensors.index.json"
    if index.is_file():
        try:
            metadata = json.loads(index.read_text())
            shards = set(metadata["weight_map"].values())
            return bool(shards) and all(
                (p / name).is_file() and (p / name).stat().st_size > 0
                for name in shards
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False
    standalone = p / "model.safetensors"
    return standalone.is_file() and standalone.stat().st_size > 0


def ensure_checkpoint_space(reference_checkpoint, destination, reserve_gib=4):
    """Fail before costly training if a complete model checkpoint cannot fit."""
    reference = Path(reference_checkpoint)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = sum(
        item.stat().st_size for item in reference.glob("*.safetensors")
        if item.is_file()
    )
    if not expected_bytes:
        raise FileNotFoundError(
            f"No model safetensors in reference checkpoint: {reference}"
        )
    required = expected_bytes + reserve_gib * 2**30
    free = shutil.disk_usage(destination.parent).free
    print(
        f"[disk] available={free / 2**30:.1f} GiB; "
        f"estimated_needed={required / 2**30:.1f} GiB "
        f"for {destination}", flush=True,
    )
    if free < required:
        raise OSError(
            "Insufficient disk for saving this BF16 7B checkpoint. "
            f"Free at least {(required-free)/2**30:.1f} more GiB "
            "(prefer additional headroom), or increase Pod disk storage. "
            "Do not delete the EN anchor or completed experiment results."
        )


def load_model(path):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    return model


def memory_report(stage):
    print(
        f"[memory/{stage}] allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB "
        f"reserved={torch.cuda.memory_reserved()/2**30:.2f} GiB "
        f"peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
        flush=True,
    )


def train_stage(model, blocks, args):
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError("Install bitsandbytes>=0.45 first.") from exc
    klass = (
        bnb.optim.AdamW8bit if args.optimizer == "adamw8bit"
        else bnb.optim.PagedAdamW8bit
    )
    optimizer = klass(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(blocks), batch_size=args.micro_batch,
                        shuffle=False, pin_memory=True)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    updates, tokens = 0, 0
    for i, (x,) in enumerate(loader):
        if args.max_optimizer_steps and updates >= args.max_optimizer_steps:
            break
        x = x.to("cuda", non_blocking=True)
        out = model(input_ids=x, labels=x, use_cache=False)
        (out.loss / args.grad_accum).backward()
        tokens += x.numel()
        if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if updates == 1 or updates % args.log_every == 0:
                print(f"[train] step={updates} tokens={tokens} "
                      f"loss={out.loss.detach().item():.5f}", flush=True)
                memory_report("train")
        del out
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return {"tokens_seen": tokens, "optimizer_steps": updates}


def protocol_args(args):
    keys = [
        "model_name", "seed", "old_language", "new_language",
        "new_train_fraction", "lr", "weight_decay", "micro_batch",
        "grad_accum", "eval_max_blocks", "eval_batch",
        "optimizer", "max_optimizer_steps",
    ]
    return {key: getattr(args, key) for key in keys}


def save_model(model, tokenizer, path):
    """Save to a temporary sibling and expose only a complete checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing directory: {path}. "
            "Inspect and manually remove a FAILED partial checkpoint first."
        )
    # Full BF16 model weights; preflight for one entire save plus headroom.
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    required = param_bytes + 4 * 2**30
    available = shutil.disk_usage(path.parent).free
    if available < required:
        raise OSError(
            f"Not enough free disk to save {path}: "
            f"{available / 2**30:.1f} GiB free, "
            f"at least {required / 2**30:.1f} GiB needed."
        )
    temp = path.parent / f".{path.name}.saving-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        model.save_pretrained(
            temp, safe_serialization=True, max_shard_size="4GB"
        )
        tokenizer.save_pretrained(temp)
        if not complete_checkpoint(temp):
            raise RuntimeError(f"New checkpoint is incomplete: {temp}")
        if path.exists():
            path.rmdir()  # only succeeds for an empty directory
        temp.rename(path)
    except BaseException:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def measure(model, old_val, new_val, eval_batch):
    model.gradient_checkpointing_disable()
    device = torch.device("cuda")
    return {
        "old": evaluate(model, old_val, eval_batch, device, True),
        "new": evaluate(model, new_val, eval_batch, device, True),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--new_train_fraction", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--eval_batch", type=int, default=1)
    ap.add_argument("--eval_max_blocks", type=int, default=128)
    ap.add_argument("--optimizer", choices=["adamw8bit", "paged_adamw8bit"],
                    default="adamw8bit")
    ap.add_argument("--max_optimizer_steps", type=int, default=0,
                    help="For smoke testing only; 0 trains all blocks.")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    if not (0 < args.new_train_fraction <= 1):
        ap.error("--new_train_fraction must be in (0,1].")
    if min(args.micro_batch, args.grad_accum, args.eval_batch) < 1:
        ap.error("All batch sizes must be positive.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires A100-class CUDA GPU with BF16.")
    if torch.cuda.get_device_properties(0).total_memory < 70 * 2**30:
        raise RuntimeError("This setup targets approximately 80GB VRAM.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    anchor_dir, adapted_dir = out / "anchor", out / "adapted"
    anchor_meta = out / "anchor_summary.json"
    final_meta = out / "training_summary.json"
    if complete_checkpoint(anchor_dir) and complete_checkpoint(adapted_dir) and final_meta.is_file():
        old_summary = json.loads(final_meta.read_text())
        if old_summary.get("protocol_args") != protocol_args(args):
            raise ValueError("Existing output uses different settings; choose another out_dir.")
        print(f"[resume] already trained: {final_meta}", flush=True)
        return

    data = Path(args.data_dir)
    old_train = load_blocks(data / f"{args.old_language}_train.pt")
    new_all = load_blocks(data / f"{args.new_language}_train.pt")
    old_val = load_blocks(data / f"{args.old_language}_val.pt")
    new_val = load_blocks(data / f"{args.new_language}_val.pt")
    if args.eval_max_blocks > 0:
        old_val = old_val[:args.eval_max_blocks]
        new_val = new_val[:args.eval_max_blocks]
    g_old = torch.Generator().manual_seed(args.seed + 1000)
    old_train = old_train[torch.randperm(len(old_train), generator=g_old)]
    g_new = torch.Generator().manual_seed(args.seed + 1001)
    perm = torch.randperm(len(new_all), generator=g_new)
    n_new = max(1, round(len(new_all) * args.new_train_fraction))
    new_train = new_all[perm[:n_new]]
    print(f"[data] old={len(old_train)} new={len(new_train)} "
          f"block_len={old_train.shape[-1]} optimizer={args.optimizer}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if complete_checkpoint(anchor_dir) and anchor_meta.is_file():
        anchor_result = json.loads(anchor_meta.read_text())
        if anchor_result.get("protocol_args") != protocol_args(args):
            raise ValueError("Anchor checkpoint uses different settings.")
        print("[resume] reuse existing anchor", flush=True)
    else:
        set_seed(args.seed)
        model = load_model(args.model_name)
        memory_report("loaded")
        trained = train_stage(model, old_train, args)
        losses = measure(model, old_val, new_val, args.eval_batch)
        save_model(model, tokenizer, anchor_dir)
        anchor_result = {
            "train": trained, "losses": losses,
            "protocol_args": protocol_args(args),
        }
        anchor_meta.write_text(json.dumps(anchor_result, indent=2))
        print(f"[anchor] {losses}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Reload stage-1 checkpoint; reset RNG and optimizer at the boundary.
    set_seed(args.seed)
    model = load_model(anchor_dir)
    adapted_train = train_stage(model, new_train, args)
    adapted_losses = measure(model, old_val, new_val, args.eval_batch)
    save_model(model, tokenizer, adapted_dir)
    summary = {
        "model_name": args.model_name,
        "seed": args.seed,
        "old_language": args.old_language,
        "new_language": args.new_language,
        "optimizer": args.optimizer,
        "precision": "bf16",
        "full_parameter_finetuning": True,
        "gradient_checkpointing": True,
        "old_stage_shuffle_seed": args.seed + 1000,
        "new_stage_shuffle_seed": args.seed + 1001,
        "new_train_fraction": args.new_train_fraction,
        "old_train_blocks": len(old_train),
        "new_train_blocks": len(new_train),
        "old_tokens_seen": anchor_result["train"]["tokens_seen"],
        "new_tokens_seen": adapted_train["tokens_seen"],
        "old_optimizer_steps": anchor_result["train"]["optimizer_steps"],
        "new_optimizer_steps": adapted_train["optimizer_steps"],
        "anchor_old_loss": anchor_result["losses"]["old"],
        "anchor_new_loss": anchor_result["losses"]["new"],
        "adapted_old_loss": adapted_losses["old"],
        "adapted_new_loss": adapted_losses["new"],
        "forgetting": adapted_losses["old"] - anchor_result["losses"]["old"],
        "new_language_gain": anchor_result["losses"]["new"] - adapted_losses["new"],
        "anchor_checkpoint": str(anchor_dir),
        "adapted_checkpoint": str(adapted_dir),
        "protocol_args": protocol_args(args),
        "caveat": "8-bit optimizer states are not numerically identical to reference AdamW.",
    }
    final_meta.write_text(json.dumps(summary, indent=2))
    print(f"[done] forgetting={summary['forgetting']:+.6f} "
          f"new_gain={summary['new_language_gain']:+.6f}", flush=True)
    memory_report("done")


if __name__ == "__main__":
    main()
