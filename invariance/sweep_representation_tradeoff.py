import argparse
import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmark_representation_methods import (
    benchmark_all_languages,
    leave_one_out_accuracy,
    evaluate_task_by_language,
    fit_task_head_on_representation,
    fresh_language_probe_accuracy,
    make_split_indices,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_control_representation(method, x, train_idx, proj_dim, seed):
    """Fit a non-learned control transform using TRAIN examples only."""
    if method == "raw":
        return x, {"dim": int(x.shape[1])}

    if method == "pca":
        mean = x[train_idx].mean(dim=0, keepdim=True)
        xc = x[train_idx] - mean
        q = min(proj_dim, x.shape[1], xc.shape[0])
        # torch.pca_lowrank is much cheaper than a full SVD for this setting.
        _, _, v = torch.pca_lowrank(xc, q=q, center=False)
        z = (x - mean) @ v[:, :q]
        return z, {"dim": int(q)}

    if method == "randproj":
        set_seed(seed)
        q = min(proj_dim, x.shape[1])
        r = torch.randn(x.shape[1], q, device=x.device, dtype=x.dtype)
        qmat, _ = torch.linalg.qr(r, mode="reduced")
        z = x @ qmat[:, :q]
        return z, {"dim": int(q)}

    raise ValueError(method)


def eval_control_all(
    method,
    x,
    labels,
    language_ids,
    train_idx,
    test_idx,
    id_to_lang,
    *,
    proj_dim,
    epochs,
    lr,
    weight_decay,
    lang_probe_epochs,
    lang_probe_lr,
    seed,
):
    z, meta = fit_control_representation(method, x, train_idx, proj_dim, seed)
    task_head = fit_task_head_on_representation(
        z,
        labels,
        train_idx,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed + 17,
    )
    with torch.no_grad():
        logits = task_head(z)

    per_lang, mean_loss, risk_var, mean_acc = evaluate_task_by_language(
        logits, labels, language_ids, test_idx, id_to_lang
    )
    lang_acc = fresh_language_probe_accuracy(
        z,
        language_ids,
        train_idx,
        test_idx,
        len(id_to_lang),
        epochs=lang_probe_epochs,
        lr=lang_probe_lr,
        weight_decay=weight_decay,
        seed=seed + 31,
    )
    return {
        "mean_task_accuracy": mean_acc,
        "mean_task_loss": mean_loss,
        "risk_variance": risk_var,
        "language_probe_accuracy": lang_acc,
        "representation_dim": meta["dim"],
        "per_language": per_lang,
    }


def eval_control_loo(
    method,
    x,
    labels,
    language_ids,
    train_mask,
    test_mask,
    heldout_id,
    *,
    proj_dim,
    epochs,
    lr,
    weight_decay,
    seed,
):
    train_idx = torch.where(train_mask & (language_ids != heldout_id))[0]
    test_idx = torch.where(test_mask & (language_ids == heldout_id))[0]
    z, _ = fit_control_representation(method, x, train_idx, proj_dim, seed)

    task_head = fit_task_head_on_representation(
        z,
        labels,
        train_idx,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed + 13,
    )
    with torch.no_grad():
        logits = task_head(z[test_idx])
    loss = torch.nn.functional.cross_entropy(logits, labels[test_idx]).item()
    acc = (logits.argmax(-1) == labels[test_idx]).float().mean().item()
    return loss, acc, len(test_idx)


def make_method_args(args, *, inlp_iters=None, dann_lambda=None):
    return SimpleNamespace(
        proj_dim=args.proj_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        irm_lambda=args.irm_lambda,
        vrex_lambda=args.vrex_lambda,
        dann_lambda=1.0 if dann_lambda is None else dann_lambda,
        lang_probe_epochs=args.lang_probe_epochs,
        lang_probe_lr=args.lang_probe_lr,
        inlp_iters=args.inlp_iters[0] if inlp_iters is None else inlp_iters,
        inlp_classifier_epochs=args.inlp_classifier_epochs,
        inlp_lr=args.inlp_lr,
    )


def append_result_rows(
    all_rows,
    loo_rows,
    *,
    config_name,
    family,
    setting,
    seed,
    layer,
    metrics,
    loo_by_lang,
):
    row = {
        "config": config_name,
        "family": family,
        "setting": setting,
        "seed": seed,
        "layer": layer,
        "mean_task_accuracy": metrics["mean_task_accuracy"],
        "mean_task_loss": metrics["mean_task_loss"],
        "risk_variance": metrics["risk_variance"],
        "language_probe_accuracy": metrics["language_probe_accuracy"],
        "representation_dim": metrics["representation_dim"],
    }
    for lang, loss, acc, n in metrics["per_language"]:
        row[f"{lang}_task_loss"] = loss
        row[f"{lang}_task_accuracy"] = acc
        row[f"{lang}_n"] = n

    loo_values = []
    for lang, loss, acc, n in loo_by_lang:
        loo_values.append(acc)
        loo_rows.append({
            "config": config_name,
            "family": family,
            "setting": setting,
            "seed": seed,
            "layer": layer,
            "heldout_language": lang,
            "heldout_task_loss": loss,
            "heldout_task_accuracy": acc,
            "n_test": n,
        })

    row["mean_loo_accuracy"] = float(np.mean(loo_values))
    row["std_loo_across_languages"] = float(np.std(loo_values))
    all_rows.append(row)

    print(
        f"{config_name:<16} seed={seed} "
        f"task={row['mean_task_accuracy']:.4f} "
        f"lang={row['language_probe_accuracy']:.4f} "
        f"loo={row['mean_loo_accuracy']:.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_file", required=True)
    ap.add_argument("--out_dir", default="invariance_analysis/tradeoff_sweeps")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--irm_lambda", type=float, default=1.0)
    ap.add_argument("--vrex_lambda", type=float, default=10.0)
    ap.add_argument(
        "--dann_lambdas", type=float, nargs="+",
        default=[0.1, 1.0, 5.0, 10.0, 20.0],
    )
    ap.add_argument(
        "--inlp_iters", type=int, nargs="+",
        default=[8, 16, 32, 64],
    )
    ap.add_argument("--lang_probe_epochs", type=int, default=150)
    ap.add_argument("--lang_probe_lr", type=float, default=1e-2)
    ap.add_argument("--inlp_classifier_epochs", type=int, default=100)
    ap.add_argument("--inlp_lr", type=float, default=1e-2)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    payload = torch.load(args.features_file, map_location="cpu")
    if str(args.layer) not in payload["features"]:
        raise ValueError(f"Layer {args.layer} not found in {args.features_file}")

    x = payload["features"][str(args.layer)].float().to(device)
    labels = payload["labels"].long().to(device)
    languages = payload["languages"]
    splits = payload["splits"]

    unique_langs = sorted(set(languages))
    lang_to_id = {lang: i for i, lang in enumerate(unique_langs)}
    id_to_lang = {i: lang for lang, i in lang_to_id.items()}
    language_ids = torch.tensor(
        [lang_to_id[v] for v in languages], dtype=torch.long, device=device
    )
    train_idx = make_split_indices(splits, "probe_train").to(device)
    test_idx = make_split_indices(splits, "probe_test").to(device)
    train_mask = torch.tensor(
        [s == "probe_train" for s in splits], dtype=torch.bool, device=device
    )
    test_mask = torch.tensor(
        [s == "probe_test" for s in splits], dtype=torch.bool, device=device
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    loo_rows = []

    for seed in args.seeds:
        paired_seed = seed + args.layer * 1009

        # Controls: raw hidden state, PCA-64, and random orthogonal projection.
        for control in ["raw", "pca", "randproj"]:
            metrics = eval_control_all(
                control, x, labels, language_ids, train_idx, test_idx, id_to_lang,
                proj_dim=args.proj_dim,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lang_probe_epochs=args.lang_probe_epochs,
                lang_probe_lr=args.lang_probe_lr,
                seed=paired_seed,
            )
            loo = []
            for heldout_id, heldout_lang in id_to_lang.items():
                loss, acc, n = eval_control_loo(
                    control, x, labels, language_ids, train_mask, test_mask,
                    heldout_id,
                    proj_dim=args.proj_dim,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    seed=paired_seed + heldout_id * 53,
                )
                loo.append((heldout_lang, loss, acc, n))
            append_result_rows(
                all_rows, loo_rows,
                config_name=control if control == "raw" else f"{control}{args.proj_dim}",
                family=control,
                setting=args.proj_dim if control != "raw" else x.shape[1],
                seed=seed,
                layer=args.layer,
                metrics=metrics,
                loo_by_lang=loo,
            )

        # Fixed objective baselines.
        for method in ["erm", "irm", "vrex"]:
            margs = make_method_args(args)
            metrics = benchmark_all_languages(
                method, x, labels, language_ids, train_idx, test_idx,
                id_to_lang, margs, paired_seed,
            )
            loo = []
            for heldout_id, heldout_lang in id_to_lang.items():
                loss, acc, n = leave_one_out_accuracy(
                    method, x, labels, language_ids, train_mask, test_mask,
                    heldout_id, margs, paired_seed + heldout_id * 53,
                )
                loo.append((heldout_lang, loss, acc, n))
            append_result_rows(
                all_rows, loo_rows,
                config_name=method,
                family=method,
                setting=1.0 if method == "irm" else (10.0 if method == "vrex" else 0.0),
                seed=seed,
                layer=args.layer,
                metrics=metrics,
                loo_by_lang=loo,
            )

        # DANN adversarial-strength sweep.
        for lam in args.dann_lambdas:
            margs = make_method_args(args, dann_lambda=lam)
            metrics = benchmark_all_languages(
                "dann", x, labels, language_ids, train_idx, test_idx,
                id_to_lang, margs, paired_seed,
            )
            loo = []
            for heldout_id, heldout_lang in id_to_lang.items():
                loss, acc, n = leave_one_out_accuracy(
                    "dann", x, labels, language_ids, train_mask, test_mask,
                    heldout_id, margs, paired_seed + heldout_id * 53,
                )
                loo.append((heldout_lang, loss, acc, n))
            append_result_rows(
                all_rows, loo_rows,
                config_name=f"dann_lam{lam:g}",
                family="dann",
                setting=lam,
                seed=seed,
                layer=args.layer,
                metrics=metrics,
                loo_by_lang=loo,
            )

        # INLP removal-strength sweep.
        for iters in args.inlp_iters:
            margs = make_method_args(args, inlp_iters=iters)
            metrics = benchmark_all_languages(
                "inlp", x, labels, language_ids, train_idx, test_idx,
                id_to_lang, margs, paired_seed,
            )
            loo = []
            for heldout_id, heldout_lang in id_to_lang.items():
                loss, acc, n = leave_one_out_accuracy(
                    "inlp", x, labels, language_ids, train_mask, test_mask,
                    heldout_id, margs, paired_seed + heldout_id * 53,
                )
                loo.append((heldout_lang, loss, acc, n))
            append_result_rows(
                all_rows, loo_rows,
                config_name=f"inlp_iter{iters}",
                family="inlp",
                setting=iters,
                seed=seed,
                layer=args.layer,
                metrics=metrics,
                loo_by_lang=loo,
            )

    all_fields = sorted({k for r in all_rows for k in r.keys()})
    with (out / "all_seed_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        w.writerows(all_rows)

    loo_fields = [
        "config", "family", "setting", "seed", "layer",
        "heldout_language", "heldout_task_loss", "heldout_task_accuracy", "n_test",
    ]
    with (out / "loo_seed_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=loo_fields)
        w.writeheader()
        w.writerows(loo_rows)

    aggregate = []
    for config in sorted(set(r["config"] for r in all_rows)):
        rows = [r for r in all_rows if r["config"] == config]
        aggregate.append({
            "config": config,
            "family": rows[0]["family"],
            "setting": rows[0]["setting"],
            "n_seeds": len(rows),
            "task_acc_mean": float(np.mean([r["mean_task_accuracy"] for r in rows])),
            "task_acc_std": float(np.std([r["mean_task_accuracy"] for r in rows])),
            "lang_acc_mean": float(np.mean([r["language_probe_accuracy"] for r in rows])),
            "lang_acc_std": float(np.std([r["language_probe_accuracy"] for r in rows])),
            "loo_acc_mean": float(np.mean([r["mean_loo_accuracy"] for r in rows])),
            "loo_acc_std": float(np.std([r["mean_loo_accuracy"] for r in rows])),
        })

    aggregate.sort(key=lambda r: r["loo_acc_mean"], reverse=True)
    agg_fields = [
        "config", "family", "setting", "n_seeds",
        "task_acc_mean", "task_acc_std",
        "lang_acc_mean", "lang_acc_std",
        "loo_acc_mean", "loo_acc_std",
    ]
    with (out / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        w.writerows(aggregate)

    meta = {
        "features_file": args.features_file,
        "layer": args.layer,
        "seeds": args.seeds,
        "proj_dim": args.proj_dim,
        "dann_lambdas": args.dann_lambdas,
        "inlp_iters": args.inlp_iters,
        "note": "All learned methods use paired seeds within each experimental seed.",
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== Aggregate ranking by held-out-language transfer ===")
    for r in aggregate:
        print(
            f"{r['config']:<16} "
            f"LOO={r['loo_acc_mean']:.4f}±{r['loo_acc_std']:.4f} "
            f"TASK={r['task_acc_mean']:.4f}±{r['task_acc_std']:.4f} "
            f"LANG={r['lang_acc_mean']:.4f}±{r['lang_acc_std']:.4f}"
        )
    print(f"\nSaved sweep results to {out}")


if __name__ == "__main__":
    main()
