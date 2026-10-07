#!/usr/bin/env python3
"""Train new-language adaptation with targeted subspace interventions.

Two intervention families:
1) projected_hidden_preservation:
   L = L_new + lambda * ||(h-h_anchor) Q||^2 / (anchor projected energy + eps)

2) activation_gradient_shielding:
   on the selected transformer block output, backpropagated activation gradient
   is replaced by g - alpha * (gQ)Q^T.

The script evaluates retention/plasticity after adaptation and is intentionally
kept separate from the existing layer-preservation experiments.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.random_layer_freeze_pilot.run_pilot import (  # noqa: E402
    get_layers,
    load_model,
)
from invariance.train_sequence import evaluate, load_blocks, make_loader  # noqa: E402


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_basis(path, subspace, layer):
    payload = torch.load(path, map_location="cpu")

    # Detailed Step-2 subspace artifact.
    if "subspaces" in payload:
        if subspace not in payload["subspaces"]:
            raise KeyError(
                f"{subspace!r} not found. Available: {list(payload['subspaces'])}"
            )
        q = payload["subspaces"][subspace].float()
        artifact_layer = int(payload.get("layer", layer))
        if artifact_layer != layer:
            raise ValueError(
                f"Basis artifact layer={artifact_layer}, requested layer={layer}"
            )
        return q

    # Gradient subspace artifact.
    if "layers" in payload and str(layer) in payload["layers"]:
        item = payload["layers"][str(layer)]
        key = subspace
        if key not in item:
            raise KeyError(
                f"{subspace!r} not in gradient layer artifact. "
                f"Available: {list(item)}"
            )
        return item[key].float()

    # Functional atlas per-layer basis artifact.
    if "anchor_subspaces" in payload:
        if subspace not in payload["anchor_subspaces"]:
            raise KeyError(
                f"{subspace!r} not found. "
                f"Available: {list(payload['anchor_subspaces'])}"
            )
        return payload["anchor_subspaces"][subspace].float()

    raise ValueError(f"Unrecognized basis artifact structure: {path}")


def projected_preservation_loss(h, h_anchor, q, eps):
    delta = h.float() - h_anchor.float()
    proj_delta = delta @ q
    proj_anchor = h_anchor.float() @ q
    raw = proj_delta.pow(2).mean()
    anchor_energy = proj_anchor.pow(2).mean().detach()
    normalized = raw / (anchor_energy + eps)
    return normalized, raw.detach(), anchor_energy


def train_condition(
    model,
    anchor,
    blocks,
    *,
    mode,
    layer_no,
    q,
    strength,
    lr,
    weight_decay,
    micro_batch,
    grad_accum,
    device,
    use_bf16,
    normalization_eps,
):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    opt.zero_grad(set_to_none=True)

    layer = get_layers(model)[layer_no - 1]
    q_gpu = q.to(device=device, dtype=torch.float32)

    hook_handle = None
    if mode == "activation_gradient_shielding":
        alpha = float(strength)

        def forward_hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output

            def grad_hook(g):
                gf = g.float()
                projected = (gf @ q_gpu) @ q_gpu.T
                return (gf - alpha * projected).to(g.dtype)

            h.register_hook(grad_hook)
            return output

        hook_handle = layer.register_forward_hook(forward_hook)

    tokens = 0
    sum_lm = 0.0
    sum_pres = 0.0
    sum_raw = 0.0
    sum_anchor_energy = 0.0
    n_batches = 0

    try:
        loader = make_loader(blocks, micro_batch)
        for bi, (x,) in enumerate(loader):
            x = x.to(device, non_blocking=True)

            h_anchor = None
            if mode == "projected_hidden_preservation":
                with torch.no_grad():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=use_bf16,
                    ):
                        aout = anchor(
                            input_ids=x,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                    h_anchor = aout.hidden_states[layer_no].detach()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                out = model(
                    input_ids=x,
                    labels=x,
                    output_hidden_states=(mode == "projected_hidden_preservation"),
                    use_cache=False,
                )
                lm_loss = out.loss

            if mode == "projected_hidden_preservation":
                pres, raw, anchor_energy = projected_preservation_loss(
                    out.hidden_states[layer_no],
                    h_anchor,
                    q_gpu,
                    normalization_eps,
                )
                total = lm_loss + strength * pres
                sum_pres += float(pres.detach())
                sum_raw += float(raw)
                sum_anchor_energy += float(anchor_energy)
            else:
                total = lm_loss

            (total / grad_accum).backward()
            sum_lm += float(lm_loss.detach())
            n_batches += 1
            tokens += x.numel()

            if (bi + 1) % grad_accum == 0 or bi + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return {
        "tokens_seen": tokens,
        "mean_lm_loss": sum_lm / max(n_batches, 1),
        "mean_projected_preservation": sum_pres / max(n_batches, 1),
        "mean_projected_raw_mse": sum_raw / max(n_batches, 1),
        "mean_projected_anchor_energy": sum_anchor_energy / max(n_batches, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--anchor_checkpoint",
        default="invariance_runs/sequence_seed0/en__zh/stage1_en",
    )
    ap.add_argument(
        "--basis_file",
        default="mechanism_runs/step2_layer20_subspaces_v2/layer20_subspaces.pt",
    )
    ap.add_argument(
        "--subspaces",
        nargs="+",
        default=["transfer", "isr_cov", "isr_multiclass", "vicreg", "drift"],
    )
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--train_fraction", type=float, default=0.20)
    ap.add_argument("--eval_max_blocks", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normalization_eps", type=float, default=1e-8)
    ap.add_argument(
        "--preservation_lambdas",
        nargs="+",
        type=float,
        default=[5.0, 20.0],
    )
    ap.add_argument(
        "--shield_alphas",
        nargs="+",
        type=float,
        default=[0.5, 1.0],
    )
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=["projected_hidden_preservation", "activation_gradient_shielding"],
        default=["projected_hidden_preservation", "activation_gradient_shielding"],
    )
    ap.add_argument("--save_checkpoints", action="store_true")
    ap.add_argument(
        "--out_dir",
        default="mechanism_runs/advanced_functional_diagnostics_v1/targeted_interventions",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if not 0 < args.train_fraction <= 1:
        raise ValueError("train_fraction must be in (0,1].")

    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    new_train = load_blocks(Path(args.data_dir) / f"{args.new_language}_train.pt")
    old_val = load_blocks(Path(args.data_dir) / f"{args.old_language}_val.pt")
    new_val = load_blocks(Path(args.data_dir) / f"{args.new_language}_val.pt")

    if args.eval_max_blocks > 0:
        old_val = old_val[: args.eval_max_blocks]
        new_val = new_val[: args.eval_max_blocks]

    g = torch.Generator().manual_seed(args.seed + 1001)
    perm = torch.randperm(len(new_train), generator=g)
    n = max(1, round(len(new_train) * args.train_fraction))
    new_train = new_train[perm[:n]]

    anchor = load_model(args.anchor_checkpoint, device, use_bf16)
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)

    anchor_old = evaluate(anchor, old_val, args.eval_batch, device, use_bf16)
    anchor_new = evaluate(anchor, new_val, args.eval_batch, device, use_bf16)
    print(
        f"[anchor] old={anchor_old:.6f} new={anchor_new:.6f} "
        f"train_blocks={len(new_train)}"
    )

    tok = AutoTokenizer.from_pretrained(args.anchor_checkpoint, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    jobs = []
    for subspace in args.subspaces:
        q = load_basis(args.basis_file, subspace, args.layer)
        q, _ = torch.linalg.qr(q.float(), mode="reduced")

        if "projected_hidden_preservation" in args.modes:
            for lam in args.preservation_lambdas:
                jobs.append(
                    ("projected_hidden_preservation", subspace, q, float(lam))
                )
        if "activation_gradient_shielding" in args.modes:
            for alpha in args.shield_alphas:
                jobs.append(
                    ("activation_gradient_shielding", subspace, q, float(alpha))
                )

    rows = []
    for ji, (mode, subspace, q, strength) in enumerate(jobs, 1):
        print(
            f"\n=== [{ji}/{len(jobs)}] {mode} "
            f"subspace={subspace} rank={q.shape[1]} strength={strength:g} ==="
        )
        seed_all(args.seed)
        model = load_model(args.anchor_checkpoint, device, use_bf16)

        stats = train_condition(
            model,
            anchor,
            new_train,
            mode=mode,
            layer_no=args.layer,
            q=q,
            strength=strength,
            lr=args.lr,
            weight_decay=args.weight_decay,
            micro_batch=args.micro_batch,
            grad_accum=args.grad_accum,
            device=device,
            use_bf16=use_bf16,
            normalization_eps=args.normalization_eps,
        )

        post_old = evaluate(model, old_val, args.eval_batch, device, use_bf16)
        post_new = evaluate(model, new_val, args.eval_batch, device, use_bf16)

        forgetting = post_old - anchor_old
        new_gain = anchor_new - post_new

        row = {
            "mode": mode,
            "subspace": subspace,
            "rank": int(q.shape[1]),
            "strength": strength,
            "layer": args.layer,
            "forgetting": forgetting,
            "new_language_gain": new_gain,
            "anchor_old_loss": anchor_old,
            "anchor_new_loss": anchor_new,
            "post_old_loss": post_old,
            "post_new_loss": post_new,
            **stats,
        }
        rows.append(row)

        print(
            f"forget={forgetting:+.6f} gain={new_gain:+.6f} "
            f"train_lm={stats['mean_lm_loss']:.6f}"
        )

        if args.save_checkpoints:
            tag = f"{mode}__{subspace}__{strength:g}".replace(".", "p")
            d = out / "checkpoints" / tag
            d.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(d)
            tok.save_pretrained(d)

        with (out / "results_partial.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        del model
        torch.cuda.empty_cache()

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    (out / "manifest.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {out / 'results.csv'}")


if __name__ == "__main__":
    main()
