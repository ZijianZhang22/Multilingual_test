#!/usr/bin/env python3
"""Measure low-rank/spectral hidden drift and functional output drift.

This script is independent of the main retention_subspace_mechanism pipeline.
It compares an anchor checkpoint with an adapted checkpoint on the SAME LM
blocks and reports, for every requested layer:
  - total relative hidden drift,
  - drift covariance spectrum,
  - participation-ratio effective dimension,
  - entropy effective rank,
  - stable rank,
  - rank required for 50/80/90/95/99% drift energy.

At the model output it reports:
  - old/new LM losses,
  - anchor->adapted and adapted->anchor KL at sampled token positions,
  - top-1 token agreement.

The KL is evaluated only at sampled positions to avoid materializing
full-vocabulary logits for every token.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.random_layer_freeze_pilot.run_pilot import load_model  # noqa: E402
from invariance.train_sequence import evaluate, load_blocks, make_loader  # noqa: E402


def get_backbone(model):
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "base_model"):
        return model.base_model
    raise ValueError("Cannot locate transformer backbone.")


def spectral_summary(evals):
    evals = evals.float().clamp_min(0)
    evals = torch.sort(evals, descending=True).values
    total = evals.sum().clamp_min(1e-30)
    p = evals / total

    participation = (total * total) / evals.pow(2).sum().clamp_min(1e-30)
    entropy = -(p[p > 0] * p[p > 0].log()).sum()
    entropy_rank = entropy.exp()
    stable_rank = total / evals.max().clamp_min(1e-30)

    cdf = torch.cumsum(p, dim=0)

    def rank_for(frac):
        idx = torch.searchsorted(
            cdf,
            torch.tensor(frac, dtype=cdf.dtype, device=cdf.device),
        )
        return int(min(int(idx.item()) + 1, len(cdf)))

    return {
        "participation_ratio": float(participation),
        "entropy_effective_rank": float(entropy_rank),
        "stable_rank": float(stable_rank),
        "rank50": rank_for(0.50),
        "rank80": rank_for(0.80),
        "rank90": rank_for(0.90),
        "rank95": rank_for(0.95),
        "rank99": rank_for(0.99),
        "top1_fraction": float(p[:1].sum()),
        "top8_fraction": float(p[:8].sum()),
        "top16_fraction": float(p[:16].sum()),
        "top32_fraction": float(p[:32].sum()),
        "top64_fraction": float(p[:64].sum()),
        "top128_fraction": float(p[:128].sum()),
    }


@torch.no_grad()
def analyze_language(
    anchor,
    adapted,
    blocks,
    *,
    layers,
    batch_size,
    device,
    use_bf16,
    token_stride,
):
    anchor_backbone = get_backbone(anchor)
    adapted_backbone = get_backbone(adapted)
    head_anchor = anchor.get_output_embeddings()
    head_adapted = adapted.get_output_embeddings()

    grams = {}
    anchor_energy = {}
    delta_energy = {}
    n_hidden_scalars = {}
    for layer in layers:
        grams[layer] = None
        anchor_energy[layer] = 0.0
        delta_energy[layer] = 0.0
        n_hidden_scalars[layer] = 0

    kl_ab_sum = 0.0
    kl_ba_sum = 0.0
    top1_same = 0
    n_logit_positions = 0

    for batch_i, (x,) in enumerate(make_loader(blocks, batch_size), 1):
        x = x.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            out_a = anchor_backbone(
                input_ids=x,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            out_b = adapted_backbone(
                input_ids=x,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        for layer in layers:
            ha = out_a.hidden_states[layer].float()
            hb = out_b.hidden_states[layer].float()
            delta = hb - ha
            flat = delta.reshape(-1, delta.shape[-1])
            gram = flat.T @ flat

            gram_cpu = gram.detach().cpu()
            if grams[layer] is None:
                grams[layer] = gram_cpu
            else:
                grams[layer].add_(gram_cpu)

            anchor_energy[layer] += float(ha.pow(2).sum())
            delta_energy[layer] += float(delta.pow(2).sum())
            n_hidden_scalars[layer] += ha.numel()

        # Sample final hidden states before the LM head.
        t = out_a.last_hidden_state.shape[1]
        positions = torch.arange(
            max(token_stride - 1, 0),
            t,
            token_stride,
            device=device,
        )
        if positions.numel() == 0:
            positions = torch.tensor([t - 1], device=device)

        ha_last = out_a.last_hidden_state[:, positions, :]
        hb_last = out_b.last_hidden_state[:, positions, :]
        logits_a = head_anchor(ha_last).float()
        logits_b = head_adapted(hb_last).float()

        logpa = torch.log_softmax(logits_a, dim=-1)
        logpb = torch.log_softmax(logits_b, dim=-1)
        pa = logpa.exp()
        pb = logpb.exp()

        kl_ab = (pa * (logpa - logpb)).sum(dim=-1)
        kl_ba = (pb * (logpb - logpa)).sum(dim=-1)
        kl_ab_sum += float(kl_ab.sum())
        kl_ba_sum += float(kl_ba.sum())

        pred_a = logits_a.argmax(dim=-1)
        pred_b = logits_b.argmax(dim=-1)
        top1_same += int((pred_a == pred_b).sum())
        n_logit_positions += pred_a.numel()

        if batch_i % 10 == 0:
            print(f"  processed batches={batch_i}", flush=True)

        del out_a, out_b, logits_a, logits_b, logpa, logpb, pa, pb

    rows = []
    spectra = {}
    for layer in layers:
        gram = grams[layer].double()
        evals = torch.linalg.eigvalsh(gram).clamp_min(0)
        spec = spectral_summary(evals)
        spectra[str(layer)] = torch.sort(evals.float(), descending=True).values

        rel_rms = math.sqrt(
            delta_energy[layer] / max(anchor_energy[layer], 1e-30)
        )
        rows.append(
            {
                "layer": layer,
                "relative_hidden_drift_rms": rel_rms,
                "delta_energy": delta_energy[layer],
                "anchor_energy": anchor_energy[layer],
                **spec,
            }
        )

    output = {
        "kl_anchor_to_adapted": kl_ab_sum / max(n_logit_positions, 1),
        "kl_adapted_to_anchor": kl_ba_sum / max(n_logit_positions, 1),
        "symmetric_kl": (
            kl_ab_sum + kl_ba_sum
        ) / max(2 * n_logit_positions, 1),
        "top1_agreement": top1_same / max(n_logit_positions, 1),
        "n_sampled_positions": n_logit_positions,
    }
    return rows, output, spectra


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--anchor_checkpoint",
        default="invariance_runs/sequence_seed0/en__zh/stage1_en",
    )
    ap.add_argument(
        "--adapted_checkpoint",
        default="mechanism_runs/step1_layer_lambda_sweep/checkpoints/full_ft",
    )
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--languages", nargs="+", default=["en", "zh"])
    ap.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[4, 8, 12, 16, 20, 24],
    )
    ap.add_argument("--max_blocks", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument(
        "--token_stride",
        type=int,
        default=64,
        help="Compute output KL every Nth token position.",
    )
    ap.add_argument(
        "--out_dir",
        default="mechanism_runs/advanced_functional_diagnostics_v1/spectral_output",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading anchor and adapted models...")
    anchor = load_model(args.anchor_checkpoint, device, use_bf16)
    adapted = load_model(args.adapted_checkpoint, device, use_bf16)
    anchor.eval()
    adapted.eval()

    all_rows = []
    outputs = {}
    spectra_all = {}

    for lang in args.languages:
        blocks = load_blocks(Path(args.data_dir) / f"{lang}_val.pt")
        if args.max_blocks > 0:
            blocks = blocks[: args.max_blocks]

        anchor_loss = evaluate(
            anchor, blocks, args.batch_size, device, use_bf16
        )
        adapted_loss = evaluate(
            adapted, blocks, args.batch_size, device, use_bf16
        )
        print(
            f"\n[{lang}] anchor_loss={anchor_loss:.6f} "
            f"adapted_loss={adapted_loss:.6f}"
        )

        rows, output_metrics, spectra = analyze_language(
            anchor,
            adapted,
            blocks,
            layers=args.layers,
            batch_size=args.batch_size,
            device=device,
            use_bf16=use_bf16,
            token_stride=args.token_stride,
        )
        for r in rows:
            r["language"] = lang
            r["anchor_lm_loss"] = anchor_loss
            r["adapted_lm_loss"] = adapted_loss
            r["lm_loss_change"] = adapted_loss - anchor_loss
            all_rows.append(r)

        outputs[lang] = {
            "anchor_lm_loss": anchor_loss,
            "adapted_lm_loss": adapted_loss,
            "lm_loss_change": adapted_loss - anchor_loss,
            **output_metrics,
        }
        spectra_all[lang] = spectra

        print(
            f"[{lang}] symKL={output_metrics['symmetric_kl']:.6f} "
            f"top1_agreement={output_metrics['top1_agreement']:.4f}"
        )

    write_csv(out / "spectral_drift_metrics.csv", all_rows)
    (out / "output_drift.json").write_text(
        json.dumps(outputs, indent=2),
        encoding="utf-8",
    )
    torch.save(spectra_all, out / "drift_spectra.pt")
    (out / "manifest.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(out / "spectral_drift_metrics.csv")
    print(out / "output_drift.json")
    print(out / "drift_spectra.pt")


if __name__ == "__main__":
    main()
