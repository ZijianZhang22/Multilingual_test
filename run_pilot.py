import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_blocks(path):
    obj = torch.load(path, map_location="cpu")
    return obj["input_ids"].long()


def split_source_train_val(blocks, val_fraction=0.05, split_seed=2026):
    """
    Deterministically reserve a small held-out subset for source-language diagnostics.

    The split is fixed across experimental seeds by default, so EN/ZH validation
    losses are directly comparable across branches and runs.
    """
    if not (0.0 < val_fraction < 0.5):
        raise ValueError("--source_val_fraction must be between 0 and 0.5")

    n = len(blocks)
    n_val = max(1, int(round(n * val_fraction)))
    if n_val >= n:
        raise ValueError("source validation split is too large for the prepared data")

    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n, generator=g)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return blocks[train_idx], blocks[val_idx]


def make_source_order(en, zh, branch, seed):
    n = min(len(en), len(zh))
    en, zh = en[:n], zh[:n]

    # Same EN/ZH examples in every trained branch; only schedule differs.
    g_en = torch.Generator().manual_seed(seed + 100)
    g_zh = torch.Generator().manual_seed(seed + 200)
    en = en[torch.randperm(n, generator=g_en)]
    zh = zh[torch.randperm(n, generator=g_zh)]

    if branch == "en_zh":
        return torch.cat([en, zh], dim=0)
    if branch == "zh_en":
        return torch.cat([zh, en], dim=0)
    if branch == "mix":
        both = torch.cat([en, zh], dim=0)
        g_mix = torch.Generator().manual_seed(seed + 300)
        return both[torch.randperm(len(both), generator=g_mix)]
    raise ValueError(branch)


def make_loader(blocks, batch_size):
    return DataLoader(
        TensorDataset(blocks),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )


def make_optimizer(model, lr, weight_decay, total_steps, warmup_ratio):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    return opt, sched


@torch.no_grad()
def evaluate(model, val_blocks, batch_size, device, use_bf16):
    model.eval()
    loader = make_loader(val_blocks, batch_size)
    weighted_loss = 0.0
    total_targets = 0

    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(use_bf16 and device.type == "cuda"),
        ):
            out = model(input_ids=x, labels=x, use_cache=False)
        n_targets = x.shape[0] * (x.shape[1] - 1)
        weighted_loss += float(out.loss) * n_targets
        total_targets += n_targets

    mean_loss = weighted_loss / total_targets
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    model.train()
    return mean_loss, ppl


def evaluate_source_diagnostics(
    model,
    en_val,
    zh_val,
    target_val,
    eval_batch,
    device,
    use_bf16,
    branch,
):
    """
    Evaluate each branch immediately after its source-history stage and BEFORE
    target-language training.

    This tells us whether the source treatment actually created distinguishable
    models before asking whether it changes future target-language plasticity.
    """
    en_loss, en_ppl = evaluate(model, en_val, eval_batch, device, use_bf16)
    zh_loss, zh_ppl = evaluate(model, zh_val, eval_batch, device, use_bf16)
    target_loss, target_ppl = evaluate(model, target_val, eval_batch, device, use_bf16)

    print(
        f"[{branch}] PRE-TARGET DIAGNOSTICS | "
        f"EN loss={en_loss:.4f} ppl={en_ppl:.2f} | "
        f"ZH loss={zh_loss:.4f} ppl={zh_ppl:.2f} | "
        f"TARGET loss={target_loss:.4f} ppl={target_ppl:.2f}"
    )

    return {
        "branch": branch,
        "en_val_loss": en_loss,
        "en_perplexity": en_ppl,
        "zh_val_loss": zh_loss,
        "zh_perplexity": zh_ppl,
        "target_val_loss_before_target": target_loss,
        "target_perplexity_before_target": target_ppl,
    }


def evaluate_forgetting(
    model,
    aux_eval_blocks,
    aux_baselines,
    eval_batch,
    device,
    use_bf16,
    branch,
    tokens_seen,
    requested_mark,
    forgetting_rows,
):
    """
    Measure old-language retention while the model learns the target language.

    Forgetting is defined as:
        current held-out source loss - pre-target held-out source loss

    Positive values mean forgetting; negative values mean improvement.
    """
    row = {
        "branch": branch,
        "target_tokens_seen": tokens_seen,
        "requested_mark": requested_mark,
    }

    parts = []
    for lang, blocks in aux_eval_blocks.items():
        loss, ppl = evaluate(model, blocks, eval_batch, device, use_bf16)
        baseline = aux_baselines[lang]
        forgetting = loss - baseline
        row[f"{lang}_val_loss"] = loss
        row[f"{lang}_perplexity"] = ppl
        row[f"{lang}_forgetting"] = forgetting
        parts.append(
            f"{lang.upper()} loss={loss:.4f} "
            f"forgetting={forgetting:+.4f}"
        )

    forgetting_rows.append(row)
    print(
        f"[{branch}] RETENTION target_tokens={tokens_seen:,} "
        f"(mark {requested_mark}) | " + " | ".join(parts)
    )


def train_stream(
    model,
    blocks,
    *,
    lr,
    weight_decay,
    warmup_ratio,
    micro_batch,
    grad_accum,
    device,
    use_bf16,
    eval_blocks=None,
    eval_batch=8,
    eval_token_marks=None,
    branch=None,
    stage=None,
    metrics_rows=None,
    aux_eval_blocks=None,
    aux_baselines=None,
    forgetting_rows=None,
):
    loader = make_loader(blocks, micro_batch)
    total_opt_steps = math.ceil(len(loader) / grad_accum)
    opt, sched = make_optimizer(model, lr, weight_decay, total_opt_steps, warmup_ratio)

    opt.zero_grad(set_to_none=True)
    tokens_seen = 0
    next_mark_idx = 0
    eval_token_marks = sorted(eval_token_marks or [])

    if eval_blocks is not None and eval_token_marks and eval_token_marks[0] == 0:
        val_loss, ppl = evaluate(model, eval_blocks, eval_batch, device, use_bf16)
        metrics_rows.append({
            "branch": branch,
            "stage": stage,
            "target_tokens_seen": 0,
            "requested_mark": 0,
            "val_loss": val_loss,
            "perplexity": ppl,
        })
        print(f"[{branch}] target_tokens=0 val_loss={val_loss:.4f} ppl={ppl:.2f}")
        if aux_eval_blocks is not None:
            evaluate_forgetting(
                model,
                aux_eval_blocks,
                aux_baselines,
                eval_batch,
                device,
                use_bf16,
                branch,
                0,
                0,
                forgetting_rows,
            )
        next_mark_idx = 1

    model.train()
    for i, (x,) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(use_bf16 and device.type == "cuda"),
        ):
            out = model(input_ids=x, labels=x, use_cache=False)
            loss = out.loss / grad_accum

        loss.backward()
        tokens_seen += x.numel()

        do_step = ((i + 1) % grad_accum == 0) or (i + 1 == len(loader))
        if do_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            if eval_blocks is not None:
                while (
                    next_mark_idx < len(eval_token_marks)
                    and tokens_seen >= eval_token_marks[next_mark_idx]
                ):
                    mark = eval_token_marks[next_mark_idx]
                    val_loss, ppl = evaluate(
                        model, eval_blocks, eval_batch, device, use_bf16
                    )
                    metrics_rows.append({
                        "branch": branch,
                        "stage": stage,
                        "target_tokens_seen": tokens_seen,
                        "requested_mark": mark,
                        "val_loss": val_loss,
                        "perplexity": ppl,
                    })
                    print(
                        f"[{branch}] target_tokens={tokens_seen:,} (mark {mark:,}) "
                        f"val_loss={val_loss:.4f} ppl={ppl:.2f}"
                    )
                    if aux_eval_blocks is not None:
                        evaluate_forgetting(
                            model,
                            aux_eval_blocks,
                            aux_baselines,
                            eval_batch,
                            device,
                            use_bf16,
                            branch,
                            tokens_seen,
                            mark,
                            forgetting_rows,
                        )
                    next_mark_idx += 1

    if eval_blocks is not None:
        branch_rows = [r for r in metrics_rows if r["branch"] == branch]
        last_seen = branch_rows[-1]["target_tokens_seen"] if branch_rows else -1
        if last_seen != tokens_seen:
            val_loss, ppl = evaluate(model, eval_blocks, eval_batch, device, use_bf16)
            metrics_rows.append({
                "branch": branch,
                "stage": stage,
                "target_tokens_seen": tokens_seen,
                "requested_mark": "final",
                "val_loss": val_loss,
                "perplexity": ppl,
            })
            print(
                f"[{branch}] FINAL target_tokens={tokens_seen:,} "
                f"val_loss={val_loss:.4f} ppl={ppl:.2f}"
            )
            if aux_eval_blocks is not None:
                evaluate_forgetting(
                    model,
                    aux_eval_blocks,
                    aux_baselines,
                    eval_batch,
                    device,
                    use_bf16,
                    branch,
                    tokens_seen,
                    "final",
                    forgetting_rows,
                )

    # Crucial: source optimizer state is NOT reused for target adaptation.
    del opt, sched
    return model


def write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "branch", "stage", "target_tokens_seen", "requested_mark",
        "val_loss", "perplexity"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def write_source_diagnostics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "branch",
        "en_val_loss", "en_perplexity",
        "zh_val_loss", "zh_perplexity",
        "target_val_loss_before_target", "target_perplexity_before_target",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def write_forgetting_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "branch", "target_tokens_seen", "requested_mark",
        "en_val_loss", "en_perplexity", "en_forgetting",
        "zh_val_loss", "zh_perplexity", "zh_forgetting",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data_dir", default="prepared")
    ap.add_argument("--out_dir", default="runs/seed0")
    ap.add_argument(
        "--branches", nargs="+",
        default=["base", "mix", "en_zh", "zh_en"],
        choices=["base", "mix", "en_zh", "zh_en"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--source_lr", type=float, default=2e-5)
    ap.add_argument("--target_lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument(
        "--source_val_fraction",
        type=float,
        default=0.05,
        help="Fraction of prepared EN/ZH blocks held out for pre-target diagnostics.",
    )
    ap.add_argument(
        "--source_split_seed",
        type=int,
        default=2026,
        help="Fixed seed for EN/ZH train/diagnostic split; keep constant across runs.",
    )
    ap.add_argument(
        "--eval_marks", type=int, nargs="+",
        default=[0, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000],
    )
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--save_models", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This pilot expects a CUDA GPU.")

    device = torch.device("cuda")
    use_bf16 = (not args.no_bf16) and torch.cuda.is_bf16_supported()
    data_dir = Path(args.data_dir)

    en_all = load_blocks(data_dir / "en.pt")
    zh_all = load_blocks(data_dir / "zh.pt")
    target_train = load_blocks(data_dir / "target_train.pt")
    target_val = load_blocks(data_dir / "target_val.pt")

    # Reserve fixed source-language diagnostic blocks. No data re-preparation is needed.
    en, en_val = split_source_train_val(
        en_all, args.source_val_fraction, args.source_split_seed
    )
    zh, zh_val = split_source_train_val(
        zh_all, args.source_val_fraction, args.source_split_seed + 1
    )

    # Keep source training budgets exactly matched after the holdout split.
    n_source = min(len(en), len(zh))
    en = en[:n_source]
    zh = zh[:n_source]

    print(
        "Source diagnostic split: "
        f"EN train={len(en):,} blocks / val={len(en_val):,}; "
        f"ZH train={len(zh):,} blocks / val={len(zh_val):,}"
    )

    # Fixed target order, reused by all branches within a seed.
    g_tgt = torch.Generator().manual_seed(args.seed + 999)
    target_train = target_train[torch.randperm(len(target_train), generator=g_tgt)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    all_rows = []
    diagnostic_rows = []
    forgetting_rows = []

    for branch in args.branches:
        print("\n" + "=" * 80)
        print(f"BRANCH: {branch} | seed={args.seed}")
        print("=" * 80)
        set_seed(args.seed)

        # Identical pretrained starting checkpoint for every branch.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=(torch.bfloat16 if use_bf16 else torch.float32),
            low_cpu_mem_usage=True,
        ).to(device)
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Stage 1: source history. One optimizer spans the whole source stream.
        if branch != "base":
            source_blocks = make_source_order(en, zh, branch, args.seed)
            print(
                f"Source stage: {len(source_blocks):,} blocks, "
                f"{source_blocks.numel():,} tokens"
            )
            model = train_stream(
                model,
                source_blocks,
                lr=args.source_lr,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                micro_batch=args.micro_batch,
                grad_accum=args.grad_accum,
                device=device,
                use_bf16=use_bf16,
            )

        # NEW: diagnose whether the source treatment created a measurable state
        # difference before any target-language optimization occurs.
        diagnostic = evaluate_source_diagnostics(
            model,
            en_val,
            zh_val,
            target_val,
            args.eval_batch,
            device,
            use_bf16,
            branch,
        )
        diagnostic_rows.append(diagnostic)
        write_source_diagnostics(
            out_dir / "source_diagnostics.csv", diagnostic_rows
        )

        source_baselines = {
            "en": diagnostic["en_val_loss"],
            "zh": diagnostic["zh_val_loss"],
        }

        if args.save_models:
            source_dir = out_dir / branch / "source_model"
            source_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(source_dir)
            tok.save_pretrained(source_dir)

        # Stage 2: same target stream, with a fresh optimizer/scheduler.
        print(
            f"Target stage: {len(target_train):,} blocks, "
            f"{target_train.numel():,} tokens"
        )
        model = train_stream(
            model,
            target_train,
            lr=args.target_lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            micro_batch=args.micro_batch,
            grad_accum=args.grad_accum,
            device=device,
            use_bf16=use_bf16,
            eval_blocks=target_val,
            eval_batch=args.eval_batch,
            eval_token_marks=args.eval_marks,
            branch=branch,
            stage="target",
            metrics_rows=all_rows,
            aux_eval_blocks={"en": en_val, "zh": zh_val},
            aux_baselines=source_baselines,
            forgetting_rows=forgetting_rows,
        )

        if args.save_models:
            final_dir = out_dir / branch / "target_final"
            final_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(final_dir)
            tok.save_pretrained(final_dir)

        write_metrics(out_dir / "metrics.csv", all_rows)
        write_forgetting_metrics(
            out_dir / "forgetting_metrics.csv", forgetting_rows
        )
        del model
        torch.cuda.empty_cache()

    print(f"\nDone. Target metrics: {out_dir / 'metrics.csv'}")
    print(f"Source diagnostics: {out_dir / 'source_diagnostics.csv'}")
    print(f"Forgetting metrics: {out_dir / 'forgetting_metrics.csv'}")


if __name__ == "__main__":
    main()
