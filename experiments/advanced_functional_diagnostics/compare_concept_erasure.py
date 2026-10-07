#!/usr/bin/env python3
"""Compare language-concept erasure controls: mean projection and LEACE.

The erasers are fitted on pooled XNLI anchor features, but may be evaluated
both intrinsically (fresh language/task probes) and causally by applying the
same affine map tokenwise at a transformer block output.

This is kept in a separate advanced diagnostics directory so it does not
alter the main mechanism pipeline.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.random_layer_freeze_pilot.run_pilot import (  # noqa: E402
    get_layers,
    load_model,
)
from invariance.benchmark_representation_methods import train_linear_classifier  # noqa: E402
from invariance.train_sequence import evaluate, load_blocks  # noqa: E402


def orth(q):
    q, _ = torch.linalg.qr(q.float(), mode="reduced")
    return q


def make_language_ids(languages):
    names = sorted(set(languages))
    mapping = {x: i for i, x in enumerate(names)}
    ids = torch.tensor([mapping[x] for x in languages], dtype=torch.long)
    return ids, names


def fit_mean_projection(x, y):
    """Orthogonal erasure of the span of class-mean differences."""
    x = x.float()
    mu = x.mean(dim=0)
    means = []
    for c in sorted(set(y.tolist())):
        means.append(x[y == c].mean(dim=0))
    m = torch.stack(means) - mu
    _, s, vh = torch.linalg.svd(m, full_matrices=False)
    rank = int((s > 1e-6).sum().item())
    q = vh[:rank].T if rank else torch.empty(x.shape[1], 0)
    if rank:
        q = orth(q)
    a = torch.eye(x.shape[1]) - q @ q.T
    b = mu - a @ mu
    return a, b, {
        "rank": rank,
        "singular_values": [float(v) for v in s[:rank]],
    }


def symmetric_sqrt_and_pinv(cov, eps=1e-7):
    evals, evecs = torch.linalg.eigh(cov.double())
    max_eval = evals.max().clamp_min(1e-30)
    keep = evals > eps * max_eval

    sqrt_vals = torch.zeros_like(evals)
    invsqrt_vals = torch.zeros_like(evals)
    sqrt_vals[keep] = torch.sqrt(evals[keep])
    invsqrt_vals[keep] = 1.0 / sqrt_vals[keep]

    sqrt = (evecs * sqrt_vals.unsqueeze(0)) @ evecs.T
    pinv_sqrt = (evecs * invsqrt_vals.unsqueeze(0)) @ evecs.T
    return sqrt.float(), pinv_sqrt.float(), int(keep.sum().item())


def fit_leace(x, y, eps=1e-7):
    """Closed-form LEACE affine eraser.

    r(x) = x - W^+ P_{W Sigma_XZ} W (x - mu),
    where W = (Sigma_XX^{1/2})^+.

    We return A,b such that r(x)=x A^T + b for row-vector inputs.
    """
    x = x.float()
    n, d = x.shape
    classes = sorted(set(y.tolist()))
    k = len(classes)

    mu = x.mean(dim=0)
    xc = x - mu

    z = torch.zeros(n, k, dtype=x.dtype)
    for j, c in enumerate(classes):
        z[:, j] = (y == c).float()
    zc = z - z.mean(dim=0, keepdim=True)

    cov_xx = (xc.T @ xc) / max(n - 1, 1)
    cov_xz = (xc.T @ zc) / max(n - 1, 1)

    sqrt_cov, whiten, cov_rank = symmetric_sqrt_and_pinv(cov_xx, eps=eps)
    u = whiten @ cov_xz
    uu, s, _ = torch.linalg.svd(u, full_matrices=False)
    erasure_rank = int((s > eps * s.max().clamp_min(1e-30)).sum().item())
    if erasure_rank:
        q = uu[:, :erasure_rank]
        proj = q @ q.T
    else:
        proj = torch.zeros(d, d)

    # W^+ = Sigma^(1/2) on the covariance support.
    removed_map = sqrt_cov @ proj @ whiten
    a = torch.eye(d) - removed_map
    b = mu - a @ mu

    return a, b, {
        "rank": erasure_rank,
        "covariance_rank": cov_rank,
        "cross_singular_values": [float(v) for v in s[:erasure_rank]],
    }


def apply_affine(x, a, b, strength=1.0):
    # Partial erasure interpolates between identity and the full affine eraser.
    erased = x @ a.T + b
    return x + strength * (erased - x)


def fresh_probe_accuracy(z, labels, train_idx, test_idx, n_classes, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = z.float().to(device)
    labels = labels.long().to(device)
    tr = train_idx.to(device)
    te = test_idx.to(device)
    head = train_linear_classifier(
        z[tr], labels[tr], n_classes, 100, 1e-2, 1e-4, seed
    )
    with torch.no_grad():
        pred = head(z[te]).argmax(dim=-1)
        return float((pred == labels[te]).float().mean().cpu())


def factorize_removed_map(a, tol=1e-6):
    """Factor (I-A)^T as left @ right.T for cheap tokenwise erasure."""
    rmap = torch.eye(a.shape[0], dtype=a.dtype) - a
    u, s, vh = torch.linalg.svd(rmap, full_matrices=False)
    keep = s > tol * s.max().clamp_min(1e-30)
    if not bool(keep.any()):
        return (
            torch.empty(a.shape[0], 0),
            torch.empty(a.shape[0], 0),
        )
    root = torch.sqrt(s[keep])
    # R^T = V S U^T = (V sqrt(S)) (U sqrt(S))^T
    left = vh[keep].T * root.unsqueeze(0)
    right = u[:, keep] * root.unsqueeze(0)
    return left, right


@torch.no_grad()
def evaluate_with_erasure(
    model,
    blocks,
    batch_size,
    device,
    use_bf16,
    *,
    layer_no,
    a,
    b,
    strength,
):
    layer = get_layers(model)[layer_no - 1]
    left, right = factorize_removed_map(a)
    left = left.to(device=device, dtype=torch.float32)
    right = right.to(device=device, dtype=torch.float32)
    b = b.to(device=device, dtype=torch.float32)

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        hf = h.float()
        removed = (hf @ left) @ right.T
        erased = hf - removed + b.view(1, 1, -1)
        new_h = hf + strength * (erased - hf)
        if isinstance(output, tuple):
            return (new_h.to(h.dtype), *output[1:])
        return new_h.to(h.dtype)

    handle = layer.register_forward_hook(hook)
    try:
        return evaluate(model, blocks, batch_size, device, use_bf16)
    finally:
        handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features_file",
        default="mechanism_runs/step2_layer20_subspaces_v2/features/anchor_probe.pt",
    )
    ap.add_argument(
        "--checkpoint",
        default="invariance_runs/sequence_seed0/en__zh/stage1_en",
    )
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--eval_max_blocks", type=int, default=64)
    ap.add_argument("--eval_batch", type=int, default=4)
    ap.add_argument(
        "--out_dir",
        default="mechanism_runs/advanced_functional_diagnostics_v1/concept_erasure",
    )
    args = ap.parse_args()

    payload = torch.load(args.features_file, map_location="cpu")
    x = payload["features"][str(args.layer)].float()
    lang_ids, lang_names = make_language_ids(payload["languages"])
    task_labels = payload["labels"].long()
    train_idx = torch.tensor(
        [i for i, s in enumerate(payload["splits"]) if s == "probe_train"],
        dtype=torch.long,
    )
    test_idx = torch.tensor(
        [i for i, s in enumerate(payload["splits"]) if s == "probe_test"],
        dtype=torch.long,
    )

    a_mean, b_mean, meta_mean = fit_mean_projection(
        x[train_idx], lang_ids[train_idx]
    )
    a_leace, b_leace, meta_leace = fit_leace(
        x[train_idx], lang_ids[train_idx]
    )

    methods = {
        "none": (torch.eye(x.shape[1]), torch.zeros(x.shape[1]), {"rank": 0}),
        "mean_projection": (a_mean, b_mean, meta_mean),
        "leace": (a_leace, b_leace, meta_leace),
    }

    intrinsic = []
    for method, (a, b, meta) in methods.items():
        z = apply_affine(x, a, b, strength=1.0)
        lang_acc = fresh_probe_accuracy(
            z, lang_ids, train_idx, test_idx, len(lang_names), seed=100
        )
        task_acc = fresh_probe_accuracy(
            z,
            task_labels,
            train_idx,
            test_idx,
            int(task_labels.max().item()) + 1,
            seed=101,
        )
        distortion = float((z - x).pow(2).mean().sqrt())
        intrinsic.append(
            {
                "method": method,
                "rank": meta["rank"],
                "language_probe_accuracy": lang_acc,
                "language_chance": 1.0 / len(lang_names),
                "task_probe_accuracy": task_acc,
                "rms_feature_distortion": distortion,
            }
        )
        print(
            f"[intrinsic] {method:16s} rank={meta['rank']:>3d} "
            f"lang={lang_acc:.4f} task={task_acc:.4f} "
            f"rms={distortion:.6f}"
        )

    causal = []
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_bf16 = torch.cuda.is_bf16_supported()
        model = load_model(args.checkpoint, device, use_bf16)
        model.eval()

        val = {}
        for lang in [args.old_language, args.new_language]:
            blocks = load_blocks(Path(args.data_dir) / f"{lang}_val.pt")
            if args.eval_max_blocks > 0:
                blocks = blocks[: args.eval_max_blocks]
            val[lang] = blocks

        baseline = {
            lang: evaluate(
                model, blocks, args.eval_batch, device, use_bf16
            )
            for lang, blocks in val.items()
        }

        for method in ["mean_projection", "leace"]:
            a, b, meta = methods[method]
            for strength in args.strengths:
                for lang, blocks in val.items():
                    loss = evaluate_with_erasure(
                        model,
                        blocks,
                        args.eval_batch,
                        device,
                        use_bf16,
                        layer_no=args.layer,
                        a=a,
                        b=b,
                        strength=strength,
                    )
                    causal.append(
                        {
                            "method": method,
                            "rank": meta["rank"],
                            "strength": strength,
                            "language": lang,
                            "baseline_loss": baseline[lang],
                            "erased_loss": loss,
                            "loss_delta": loss - baseline[lang],
                        }
                    )
                    print(
                        f"[causal] {method:16s} beta={strength:.2f} "
                        f"{lang} delta={loss-baseline[lang]:+.6f}"
                    )

        del model
        torch.cuda.empty_cache()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "intrinsic_erasure.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(intrinsic[0].keys()))
        w.writeheader()
        w.writerows(intrinsic)

    if causal:
        with (out / "causal_erasure.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(causal[0].keys()))
            w.writeheader()
            w.writerows(causal)

    torch.save(
        {
            "layer": args.layer,
            "language_names": lang_names,
            "mean_projection": {
                "A": a_mean,
                "b": b_mean,
                "metadata": meta_mean,
            },
            "leace": {
                "A": a_leace,
                "b": b_leace,
                "metadata": meta_leace,
            },
        },
        out / "erasers.pt",
    )
    (out / "manifest.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
