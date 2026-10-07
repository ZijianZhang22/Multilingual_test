import argparse
import csv
from pathlib import Path

import torch

from advanced_preservation_utils import (
    fit_hidden_gradient_subspace,
    load_blocks,
    shuffled_blocks,
)


def squared_norm_rows(x):
    x = x.float()
    return (x * x).sum(dim=1)


def ensure_aligned(anchor, post):
    for key in ["languages", "splits", "example_ids"]:
        if anchor.get(key) != post.get(key):
            raise ValueError(f"Anchor/post mismatch in {key}")
    if not torch.equal(anchor["labels"], post["labels"]):
        raise ValueError("Anchor/post labels do not align")


def drift_metrics(delta, q_task):
    total_e = squared_norm_rows(delta)
    task_e = squared_norm_rows(delta @ q_task)
    null_e = (total_e - task_e).clamp_min(0.0)

    d = delta.shape[1]
    k = q_task.shape[1]
    null_dim = max(d - k, 1)
    total_sum = float(total_e.sum().item())

    return {
        "total_drift_l2": float(total_e.mean().sqrt().item()),
        "task_sensitive_drift_l2": float(task_e.mean().sqrt().item()),
        "task_null_drift_l2": float(null_e.mean().sqrt().item()),
        "task_sensitive_drift_per_dim": float(task_e.mean().item()) / max(k, 1),
        "task_null_drift_per_dim": float(null_e.mean().item()) / null_dim,
        "task_sensitive_drift_fraction": (
            float(task_e.sum().item()) / total_sum if total_sum > 0 else 0.0
        ),
        "task_null_drift_fraction": (
            float(null_e.sum().item()) / total_sum if total_sum > 0 else 0.0
        ),
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Decompose anchor->post representation drift into an old-task-sensitive "
            "hidden-gradient subspace and its orthogonal null complement."
        )
    )
    ap.add_argument("--anchor_checkpoint", required=True)
    ap.add_argument("--anchor_features", required=True)
    ap.add_argument("--post_features", required=True)
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--gradient_batch_size", type=int, default=2)
    ap.add_argument("--gradient_max_batches", type=int, default=8)
    ap.add_argument("--max_blocks", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--subspace_out", default=None)
    ap.add_argument("--no_bf16", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for hidden-gradient subspace fitting")
    device = torch.device("cuda")
    use_bf16 = (not args.no_bf16) and torch.cuda.is_bf16_supported()

    old_blocks_path = Path(args.data_dir) / f"{args.old_language}_val.pt"
    if not old_blocks_path.exists():
        raise FileNotFoundError(old_blocks_path)
    blocks = load_blocks(old_blocks_path)
    blocks = shuffled_blocks(blocks, args.seed + 313, args.max_blocks)

    q_task, evals = fit_hidden_gradient_subspace(
        args.anchor_checkpoint,
        blocks,
        layer=args.layer,
        rank=args.rank,
        batch_size=args.gradient_batch_size,
        max_batches=args.gradient_max_batches,
        device=device,
        use_bf16=use_bf16,
    )
    q_task = q_task.float()

    a = torch.load(args.anchor_features, map_location="cpu")
    p = torch.load(args.post_features, map_location="cpu")
    ensure_aligned(a, p)
    layer = str(args.layer)
    if layer not in a["features"] or layer not in p["features"]:
        raise ValueError(f"Layer {args.layer} missing from anchor/post features")

    xa = a["features"][layer].float()
    xp = p["features"][layer].float()
    if xa.shape[1] != q_task.shape[0]:
        raise ValueError(
            f"Feature dim {xa.shape[1]} does not match task basis dim {q_task.shape[0]}"
        )
    delta = xp - xa

    splits = a["splits"]
    languages = a["languages"]
    rows = []

    all_idx = torch.tensor(
        [i for i, s in enumerate(splits) if s == "probe_test"], dtype=torch.long
    )
    rows.append({
        "language": "__all__",
        "n": int(all_idx.numel()),
        "task_rank": int(q_task.shape[1]),
        **drift_metrics(delta[all_idx], q_task),
    })

    for lang in sorted(set(languages)):
        idx = torch.tensor(
            [
                i for i, (l, s) in enumerate(zip(languages, splits))
                if l == lang and s == "probe_test"
            ],
            dtype=torch.long,
        )
        rows.append({
            "language": lang,
            "n": int(idx.numel()),
            "task_rank": int(q_task.shape[1]),
            **drift_metrics(delta[idx], q_task),
        })

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    subspace_out = Path(args.subspace_out) if args.subspace_out else out.with_name(
        "task_sensitive_subspace.pt"
    )
    torch.save({
        "layer": args.layer,
        "rank": int(q_task.shape[1]),
        "task_sensitive_subspace_basis": q_task,
        "eigenvalues": evals,
        "anchor_checkpoint": args.anchor_checkpoint,
        "old_language": args.old_language,
        "definition": (
            "Top eigenvectors of old-language LM-loss hidden-gradient covariance "
            "at the anchor checkpoint."
        ),
        "gradient_batch_size": args.gradient_batch_size,
        "gradient_max_batches": args.gradient_max_batches,
        "max_blocks": args.max_blocks,
    }, subspace_out)

    print("\n=== Task-sensitive vs task-null drift ===")
    for r in rows:
        print(
            f"{r['language']:<8} n={r['n']:<5} total={r['total_drift_l2']:.4f} "
            f"task={r['task_sensitive_drift_l2']:.4f} null={r['task_null_drift_l2']:.4f} "
            f"task/dim={r['task_sensitive_drift_per_dim']:.6f} "
            f"null/dim={r['task_null_drift_per_dim']:.6f}"
        )
    print(f"Saved: {out}")
    print(f"Saved task-sensitive subspace: {subspace_out}")


if __name__ == "__main__":
    main()
