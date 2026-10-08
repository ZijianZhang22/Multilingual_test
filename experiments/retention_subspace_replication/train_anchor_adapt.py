#!/usr/bin/env python3
"""Train one controlled EN->ZH replication: pretrained -> old-language anchor -> new-language adapted."""
import argparse, json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from invariance.train_sequence import evaluate, load_blocks, save_checkpoint, set_seed, train_one_language

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", default="en"); ap.add_argument("--new_language", default="zh")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--new_train_fraction", type=float, default=0.20)
    ap.add_argument("--new_data_shuffle_seed_base", type=int, default=1001)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--micro_batch", type=int, default=1); ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--eval_batch", type=int, default=4); ap.add_argument("--eval_max_blocks", type=int, default=128)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--out_dir", required=True)
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    if not 0 < args.new_train_fraction <= 1: raise ValueError("--new_train_fraction must be in (0,1].")
    device=torch.device("cuda"); use_bf16=torch.cuda.is_bf16_supported()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    anchor_dir=out/"anchor"; adapted_dir=out/"adapted"
    old_train=load_blocks(Path(args.data_dir)/f"{args.old_language}_train.pt")
    new_train_all=load_blocks(Path(args.data_dir)/f"{args.new_language}_train.pt")
    old_val=load_blocks(Path(args.data_dir)/f"{args.old_language}_val.pt")
    new_val=load_blocks(Path(args.data_dir)/f"{args.new_language}_val.pt")
    if args.eval_max_blocks>0:
        old_val=old_val[:args.eval_max_blocks]; new_val=new_val[:args.eval_max_blocks]
    g_old=torch.Generator().manual_seed(args.seed+1000)
    old_train=old_train[torch.randperm(len(old_train), generator=g_old)]
    new_shuffle_seed=args.new_data_shuffle_seed_base+args.seed
    g_new=torch.Generator().manual_seed(new_shuffle_seed)
    perm_new=torch.randperm(len(new_train_all), generator=g_new)
    n_new=max(1, round(len(new_train_all)*args.new_train_fraction))
    new_train=new_train_all[perm_new[:n_new]]
    tok=AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    set_seed(args.seed)
    model=AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache=False
    if args.gradient_checkpointing: model.gradient_checkpointing_enable()
    print(f"[setup] model={args.model_name} seed={args.seed} old_train_blocks={len(old_train)} new_train_blocks={len(new_train)} new_fraction={args.new_train_fraction:.3f}", flush=True)
    old_tokens=train_one_language(model, old_train, lr=args.lr, weight_decay=args.weight_decay, micro_batch=args.micro_batch, grad_accum=args.grad_accum, device=device, use_bf16=use_bf16)
    save_checkpoint(model, tok, anchor_dir)
    anchor_old=evaluate(model, old_val, args.eval_batch, device, use_bf16)
    anchor_new=evaluate(model, new_val, args.eval_batch, device, use_bf16)
    print(f"[anchor] old={anchor_old:.6f} new={anchor_new:.6f}", flush=True)
    set_seed(args.seed)
    new_tokens=train_one_language(model, new_train, lr=args.lr, weight_decay=args.weight_decay, micro_batch=args.micro_batch, grad_accum=args.grad_accum, device=device, use_bf16=use_bf16)
    save_checkpoint(model, tok, adapted_dir)
    adapted_old=evaluate(model, old_val, args.eval_batch, device, use_bf16)
    adapted_new=evaluate(model, new_val, args.eval_batch, device, use_bf16)
    forgetting=adapted_old-anchor_old; gain=anchor_new-adapted_new
    print(f"[adapted] old={adapted_old:.6f} new={adapted_new:.6f} forget={forgetting:+.6f} gain={gain:+.6f}", flush=True)
    summary={
        "model_name":args.model_name,"seed":args.seed,"old_language":args.old_language,"new_language":args.new_language,
        "old_stage_shuffle_seed":args.seed+1000,"new_stage_shuffle_seed":new_shuffle_seed,
        "new_train_fraction":args.new_train_fraction,"old_train_blocks":len(old_train),"new_train_blocks":len(new_train),
        "old_tokens_seen":old_tokens,"new_tokens_seen":new_tokens,"lr":args.lr,"weight_decay":args.weight_decay,
        "micro_batch":args.micro_batch,"grad_accum":args.grad_accum,
        "anchor_old_loss":anchor_old,"anchor_new_loss":anchor_new,"adapted_old_loss":adapted_old,"adapted_new_loss":adapted_new,
        "forgetting":forgetting,"new_language_gain":gain,"anchor_checkpoint":str(anchor_dir),"adapted_checkpoint":str(adapted_dir)
    }
    (out/"training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out/'training_summary.json'}", flush=True)
if __name__=="__main__": main()
